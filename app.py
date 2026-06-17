import os
import sqlite3
import datetime
import threading
import subprocess
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from PIL import Image
import io

# Custom modules (create placeholder files if missing)
try:
    from models.cnn_model import load_model, predict_image
    from utils.preprocessing import preprocess_uploaded_image
    from utils.report_generator import generate_report, generate_recommendations
except ImportError:
    print("WARNING: Custom modules missing. Using fallbacks.")
    def load_model(p): return None
    def predict_image(m, img): return ("Benign", 0.95)
    def preprocess_uploaded_image(f): return np.zeros((1,224,224,3))
    def generate_report(p, c): return "<p>Report</p>"
    def generate_recommendations(p, c): return "<p>Recommendations</p>"

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'database/predictions.db'
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('SELECT id, username FROM users WHERE id = ?', (user_id,))
    user_data = c.fetchone()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1])
    return None

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('models/saved', exist_ok=True)

training_status = {'running': False, 'progress': 0, 'message': '', 'error': None}

def init_db():
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      filename TEXT, prediction TEXT, confidence REAL,
                      report TEXT, recommendations TEXT, timestamp TEXT,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")

try:
    init_db()
except Exception as e:
    print(f"Failed to initialize database at startup: {e}")

def check_dependencies():
    missing = []
    for pkg in ['numpy', 'tensorflow', 'sklearn']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return f"Missing: {', '.join(missing)}. Run: pip install {' '.join(missing)}"
    return True

model = None
MODEL_PATH = 'models/saved/breast_cancer_model.h5'

def get_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Model load error: {e}")
            model = None
    return model

# ── AUTH ROUTES ──────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template('login.html', error='Username and password required')
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('SELECT id, username, password FROM users WHERE username = ?', (username,))
        user_data = c.fetchone()
        conn.close()
        if user_data and check_password_hash(user_data[2], password):
            user = User(user_data[0], user_data[1])
            login_user(user)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if not username or not password or not confirm_password:
            return render_template('signup.html', error='All fields required')
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')
        if len(password) < 6:
            return render_template('signup.html', error='Password must be at least 6 characters')
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        if c.fetchone():
            conn.close()
            return render_template('signup.html', error='Username already exists')
        hashed_password = generate_password_hash(password)
        try:
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            new_user_id = c.lastrowid
            conn.close()
            user = User(new_user_id, username)
            login_user(user)
            return redirect(url_for('index'))
        except Exception as e:
            conn.close()
            return render_template('signup.html', error=f'Error creating account: {str(e)}')
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ── PAGE ROUTES ───────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=current_user.username)

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html', username=current_user.username)

@app.route('/history')
@login_required
def history_page():
    return render_template('history.html', username=current_user.username)

@app.route('/training')
@login_required
def training_page():
    return render_template('training.html', username=current_user.username)

@app.route('/about')
@login_required
def about_page():
    return render_template('about.html', username=current_user.username)

# ── API ROUTES ────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        proc = preprocess_uploaded_image(filepath)
        m = get_model()
        if m is None:
            return jsonify({'error': 'Model not trained'}), 500
        pred, conf = predict_image(m, proc)
        report = generate_report(pred, conf)
        rec = generate_recommendations(pred, conf)
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO predictions (user_id, filename, prediction, confidence, report, recommendations, timestamp) VALUES (?,?,?,?,?,?,?)",
                  (current_user.id, filename, pred, conf, report, rec, ts))
        conn.commit()
        pid = c.lastrowid
        conn.close()
        return jsonify({
            'success': True,
            'prediction': pred,
            'confidence': float(conf),
            'report': report,
            'recommendations': rec,
            'image_preview': f'/uploads/{filename}',
            'id': pid,
            'timestamp': ts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
@login_required
def predict_batch():
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files'}), 400
    m = get_model()
    if m is None:
        return jsonify({'error': 'Model not trained'}), 500
    results = []
    for f in files:
        if f.filename == '':
            continue
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        try:
            proc = preprocess_uploaded_image(path)
            pred, conf = predict_image(m, proc)
            short = "Urgent" if pred == "Malignant" else "Routine"
            report = generate_report(pred, conf)
            rec = generate_recommendations(pred, conf)
            conn = sqlite3.connect(app.config['DATABASE'])
            c = conn.cursor()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO predictions (user_id, filename, prediction, confidence, report, recommendations, timestamp) VALUES (?,?,?,?,?,?,?)",
                      (current_user.id, filename, pred, conf, report, rec, ts))
            conn.commit()
            conn.close()
            results.append({'filename': filename, 'prediction': pred, 'confidence': float(conf), 'short_recommendation': short})
        except Exception as e:
            results.append({'filename': filename, 'error': str(e)})
    return jsonify({'success': True, 'results': results})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history-data')
@login_required
def get_history():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT id, filename, prediction, confidence, timestamp FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 50", (current_user.id,))
    rows = c.fetchall()
    conn.close()
    return jsonify([{'id': r[0], 'filename': r[1], 'prediction': r[2], 'confidence': r[3], 'timestamp': r[4]} for r in rows])

@app.route('/prediction/<int:pid>')
@login_required
def get_prediction_details(pid):
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT filename, prediction, confidence, report, recommendations, timestamp FROM predictions WHERE id=? AND user_id=?", (pid, current_user.id))
    row = c.fetchone()
    conn.close()
    if row:
        return jsonify({
            'filename': row[0], 'prediction': row[1], 'confidence': row[2],
            'report': row[3], 'recommendations': row[4], 'timestamp': row[5]
        })
    return jsonify({'error': 'Not found'}), 404

@app.route('/stats')
@login_required
def get_stats():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT timestamp, prediction, confidence FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 10", (current_user.id,))
    rows = c.fetchall()
    conn.close()
    stats = [{'timestamp': r[0], 'prediction': r[1], 'confidence': r[2]} for r in reversed(rows)]
    return jsonify(stats)

@app.route('/training/status')
def training_status_route():
    return jsonify(training_status)

@app.route('/training/start', methods=['POST'])
def start_training():
    global training_status
    if training_status['running']:
        return jsonify({'error': 'Training already in progress'}), 400
    dep = check_dependencies()
    if dep is not True:
        return jsonify({'error': dep}), 500
    epochs = int(request.form.get('epochs', 5))
    zip_file = request.files.get('dataset_zip')
    if not zip_file:
        return jsonify({'error': 'ZIP required'}), 400
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'training_dataset.zip')
    zip_file.save(zip_path)

    def run():
        global training_status
        try:
            training_status = {'running': True, 'progress': 10, 'message': 'Preparing...', 'error': None}
            cmd = ['python', 'train_local.py', '--zip', zip_path, '--epochs', str(epochs)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in proc.stdout:
                if 'Epoch' in line:
                    training_status['progress'] = min(90, training_status['progress'] + 1)
                training_status['message'] = line.strip()
            proc.wait()
            if proc.returncode == 0:
                training_status = {'running': False, 'progress': 100, 'message': 'Training completed!', 'error': None}
                global model
                model = None
                get_model()
            else:
                training_status = {'running': False, 'progress': 0, 'message': 'Training failed', 'error': proc.stderr.read()}
        except Exception as e:
            training_status = {'running': False, 'progress': 0, 'message': 'Error', 'error': str(e)}
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    threading.Thread(target=run).start()
    return jsonify({'status': 'started'})

@app.route('/model/info')
def model_info():
    m = get_model()
    return jsonify({'status': 'trained' if m else 'not_trained'})

@app.route('/clear-history', methods=['DELETE'])
@login_required
def clear_history():
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        conn.execute("DELETE FROM predictions WHERE user_id = ?", (current_user.id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
