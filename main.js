// ========== TAB SWITCHING (MUST WORK) ==========
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.navbar .nav-link');
    const tabContents = document.querySelectorAll('.tab-content');
    function switchTab(tabId) {
        tabContents.forEach(tc => tc.classList.remove('active'));
        const activeContent = document.getElementById(tabId + '-tab');
        if (activeContent) activeContent.classList.add('active');
        navLinks.forEach(link => link.classList.remove('active'));
        const activeLink = document.querySelector(`.navbar .nav-link[data-tab="${tabId}"]`);
        if (activeLink) activeLink.classList.add('active');
        if (tabId === 'history') loadHistory();
    }
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = link.getAttribute('data-tab');
            if (tabId) switchTab(tabId);
        });
    });
    switchTab('predict'); // default
});

// ========== SINGLE IMAGE UPLOAD ==========
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
let selectedFile = null;

if (dropZone) {
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#9b87f5'; });
    dropZone.addEventListener('dragleave', () => dropZone.style.borderColor = '#cdd5e0');
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#cdd5e0';
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });
}
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPEG or PNG)');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append('file', selectedFile);
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.success) {
            displayResult(data);
            loadHistory();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-microscope"></i> Analyze Image';
    }
});

// ========== CHARTS ==========
let probChart = null, trendChart = null;

async function renderCharts(prediction, confidence) {
    const chartSection = document.getElementById('chartSection');
    if (!chartSection) return;
    const ctxProb = document.getElementById('probChart').getContext('2d');
    if (probChart) probChart.destroy();
    probChart = new Chart(ctxProb, {
        type: 'bar',
        data: {
            labels: ['Benign', 'Malignant'],
            datasets: [{
                label: 'Probability',
                data: [prediction === 'Benign' ? confidence : 1 - confidence, prediction === 'Malignant' ? confidence : 1 - confidence],
                backgroundColor: ['#2a9d8f', '#e76f51'],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true, max: 1, title: { display: true, text: 'Probability' } } },
            plugins: { tooltip: { callbacks: { label: (ctx) => `${(ctx.raw * 100).toFixed(1)}%` } } }
        }
    });
    const resp = await fetch('/stats');
    const history = await resp.json();
    const labels = history.map(s => new Date(s.timestamp).toLocaleDateString());
    const accuracies = history.map(s => s.confidence * 100);
    const ctxTrend = document.getElementById('trendChart').getContext('2d');
    if (trendChart) trendChart.destroy();
    trendChart = new Chart(ctxTrend, {
        type: 'line',
        data: { labels, datasets: [{ label: 'Confidence (%)', data: accuracies, borderColor: '#9b87f5', backgroundColor: 'rgba(155,135,245,0.1)', fill: true, tension: 0.3, pointBackgroundColor: '#9b87f5' }] },
        options: { responsive: true, scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: 'Confidence (%)' } } }, plugins: { tooltip: { callbacks: { label: (ctx) => `${ctx.raw.toFixed(1)}%` } } } }
    });
    chartSection.style.display = 'block';
}

function displayResult(data) {
    const resultCard = document.getElementById('resultCard');
    const resultContent = document.getElementById('resultContent');
    const badgeClass = data.prediction === 'Malignant' ? 'badge-malignant' : 'badge-benign';
    const confidencePercent = (data.confidence * 100).toFixed(1);
    resultContent.innerHTML = `
        <div class="text-center mb-3"><span class="${badgeClass}"><i class="fas ${data.prediction === 'Malignant' ? 'fa-skull' : 'fa-smile'}"></i> ${data.prediction}</span></div>
        <div class="prediction-stats"><h5>Confidence Score</h5><div class="confidence-meter"><div class="confidence-fill" style="width: ${confidencePercent}%"></div></div><h3>${confidencePercent}%</h3></div>
        <div class="mt-3">${data.report}</div><div class="mt-3">${data.recommendations}</div>
        <hr><p class="text-muted"><small>Analyzed on: ${data.timestamp}</small></p>
        <button class="btn btn-outline-secondary me-2" onclick="window.open('${data.image_preview}', '_blank')"><i class="fas fa-eye"></i> View Image</button>
    `;
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth' });
    renderCharts(data.prediction, data.confidence);
    const pdfBtn = document.getElementById('downloadPdfBtn');
    if (pdfBtn) pdfBtn.onclick = () => saveReportAsPDF(data);
}

function saveReportAsPDF(data) {
    const confidencePercent = (data.confidence * 100).toFixed(1);
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
        <!DOCTYPE html><html><head><title>BreastCancerAI Report</title><meta charset="UTF-8">
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; padding: 2rem; }
            h1 { color: #9b87f5; border-bottom: 2px solid #9b87f5; }
            .badge-malignant { background: #e76f51; color: white; padding: 0.2rem 0.8rem; border-radius: 2rem; }
            .badge-benign { background: #2a9d8f; color: white; padding: 0.2rem 0.8rem; border-radius: 2rem; }
            .confidence-meter { background: #e9ecef; height: 20px; border-radius: 10px; margin: 1rem 0; width: 100%; }
            .confidence-fill { background: linear-gradient(90deg, #9b87f5, #f4a261); height: 100%; border-radius: 10px; width: ${data.confidence*100}%; }
        </style>
        </head>
        <body>
            <h1>BreastCancerAI – Diagnostic Report</h1>
            <p><strong>Filename:</strong> ${escapeHtml(data.image_preview.split('/').pop())}</p>
            <p><strong>Analysis Date:</strong> ${escapeHtml(data.timestamp)}</p>
            <p><strong>Prediction:</strong> <span class="${data.prediction === 'Malignant' ? 'badge-malignant' : 'badge-benign'}">${escapeHtml(data.prediction)}</span></p>
            <p><strong>Confidence Score:</strong> ${confidencePercent}%</p>
            <div class="confidence-meter"><div class="confidence-fill"></div></div>
            <div><h3>Pathology Report</h3>${cleanHtml(data.report)}</div>
            <div><h3>Recommendations</h3>${cleanHtml(data.recommendations)}</div>
            <hr><small>Generated by BreastCancerAI on ${new Date().toLocaleString()}</small>
        </body>
        </html>
    `);
    printWindow.document.close();
    printWindow.print();
}
function escapeHtml(str) { if (!str) return ''; return str.replace(/[&<>]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m])); }
function cleanHtml(html) { if (!html) return ''; return html.replace(/<script.*?<\/script>/gi, ''); }

// ========== BATCH MODE (CHUNKED) ==========
const batchSwitch = document.getElementById('batchModeSwitch');
const singleArea = document.getElementById('singleUploadArea');
const batchArea = document.getElementById('batchUploadArea');
if (batchSwitch) {
    batchSwitch.addEventListener('change', function() {
        if (this.checked) {
            singleArea.style.display = 'none';
            batchArea.style.display = 'block';
        } else {
            singleArea.style.display = 'block';
            batchArea.style.display = 'none';
            document.getElementById('batchResults').style.display = 'none';
            document.getElementById('batchTableBody').innerHTML = '';
        }
    });
}
const folderInput = document.getElementById('folderInput');
if (folderInput) {
    folderInput.addEventListener('change', async (e) => {
        const allFiles = Array.from(e.target.files).filter(f => /\.(jpg|jpeg|png)$/i.test(f.name));
        if (allFiles.length === 0) { alert('No valid images found.'); return; }
        const CHUNK_SIZE = 50;
        let processed = 0, allResults = [];
        const progressDiv = document.getElementById('batchProgress'), progressBar = document.getElementById('batchProgressBar'), statusP = document.getElementById('batchStatus'), resultsDiv = document.getElementById('batchResults'), tbody = document.getElementById('batchTableBody');
        progressDiv.style.display = 'block'; resultsDiv.style.display = 'none'; progressBar.style.width = '0%';
        for (let i = 0; i < allFiles.length; i += CHUNK_SIZE) {
            const chunk = allFiles.slice(i, i + CHUNK_SIZE);
            const formData = new FormData();
            chunk.forEach(file => formData.append('files[]', file));
            statusP.innerText = `Processing ${Math.min(processed + chunk.length, allFiles.length)} of ${allFiles.length}...`;
            progressBar.style.width = `${(processed / allFiles.length) * 100}%`;
            try {
                const resp = await fetch('/predict-batch', { method: 'POST', body: formData });
                const data = await resp.json();
                if (data.success) {
                    allResults = allResults.concat(data.results);
                    processed += chunk.length;
                    tbody.innerHTML = '';
                    allResults.forEach(res => {
                        tbody.insertAdjacentHTML('beforeend', `<tr><td>${res.filename}</td><td><span class="badge ${res.prediction === 'Malignant' ? 'bg-danger' : 'bg-success'}">${res.prediction}</span></td><td>${(res.confidence*100).toFixed(1)}%</td><td>${res.short_recommendation}</td></tr>`);
                    });
                    resultsDiv.style.display = 'block';
                } else throw new Error(data.error);
            } catch (err) { alert(`Error: ${err.message}`); break; }
        }
        progressBar.style.width = '100%'; statusP.innerText = `Completed! Processed ${processed} images.`;
        if (allResults.length) document.getElementById('downloadCsvBtn').onclick = () => downloadCSV(allResults);
        setTimeout(() => progressDiv.style.display = 'none', 3000);
    });
}
function downloadCSV(results) {
    let csv = "Filename,Prediction,Confidence,Recommendation\n";
    results.forEach(r => csv += `"${r.filename}",${r.prediction},${r.confidence},"${r.short_recommendation}"\n`);
    const blob = new Blob([csv], {type: 'text/csv'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'batch_report.csv';
    a.click();
}

// ========== HISTORY ==========
async function loadHistory() {
    const resp = await fetch('/history');
    const history = await resp.json();
    const tbody = document.getElementById('historyBody');
    if (!history.length) { tbody.innerHTML = '<tr><td colspan="6" class="text-center">No predictions yet.</td></tr>'; return; }
    tbody.innerHTML = history.map(item => `<tr><td>${item.id}</td><td>${item.filename}</td><td><span class="badge ${item.prediction === 'Malignant' ? 'bg-danger' : 'bg-success'}">${item.prediction}</span></td><td>${(item.confidence*100).toFixed(1)}%</td><td>${item.timestamp}</td><td><button class="btn btn-sm btn-info" onclick="viewDetails(${item.id})">Details</button></td></tr>`).join('');
}
window.viewDetails = async (id) => {
    const resp = await fetch(`/prediction/${id}`);
    const data = await resp.json();
    if (data.error) return;
    displayResult({ prediction: data.prediction, confidence: data.confidence, report: data.report, recommendations: data.recommendations, timestamp: data.timestamp, image_preview: `/uploads/${data.filename}`, success: true });
    document.querySelector('.navbar .nav-link[data-tab="predict"]').click();
};

// ========== TRAINING (ZIP) ==========
const trainForm = document.getElementById('trainForm');
if (trainForm) {
    trainForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const zipFile = document.getElementById('datasetZip').files[0];
        if (!zipFile) { alert('Please select a ZIP file.'); return; }
        const epochs = document.getElementById('epochs').value;
        const formData = new FormData();
        formData.append('dataset_zip', zipFile); formData.append('epochs', epochs);
        const startBtn = document.getElementById('startTrainBtn');
        startBtn.disabled = true; startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
        const statusDiv = document.getElementById('trainingStatus'), progressBar = document.getElementById('trainProgressBar'), messageDiv = document.getElementById('trainMessage');
        statusDiv.style.display = 'block';
        try {
            const resp = await fetch('/training/start', { method: 'POST', body: formData });
            const data = await resp.json();
            if (data.status === 'started') {
                const interval = setInterval(async () => {
                    const status = await (await fetch('/training/status')).json();
                    progressBar.style.width = `${status.progress}%`; progressBar.innerText = `${status.progress}%`; messageDiv.innerText = status.message;
                    if (!status.running) {
                        clearInterval(interval);
                        startBtn.disabled = false; startBtn.innerHTML = '<i class="fas fa-play"></i> Start Training';
                        if (status.error) {
                            document.getElementById('trainError').style.display = 'block';
                            document.getElementById('trainError').innerText = `Error: ${status.error}`;
                        } else {
                            document.getElementById('trainSuccess').style.display = 'block';
                            document.getElementById('trainSuccess').innerHTML = '<i class="fas fa-check-circle"></i> Training completed! Model is ready.';
                            setTimeout(() => document.getElementById('trainSuccess').style.display = 'none', 5000);
                        }
                    }
                }, 2000);
            } else if (data.error) alert('Error: ' + data.error);
        } catch (err) { alert('Error: ' + err.message); }
        finally { startBtn.disabled = false; startBtn.innerHTML = '<i class="fas fa-play"></i> Start Training'; }
    });
}

async function checkModelStatus() {
    const resp = await fetch('/model/info');
    const data = await resp.json();
    if (data.status !== 'trained') {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-warning alert-dismissible fade show m-3';
        alertDiv.innerHTML = `<strong>Model not trained!</strong> Please go to the Training tab and train first.<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
        document.querySelector('.main-content .container-fluid').prepend(alertDiv);
    }
}
checkModelStatus();

// ========== VIRTUAL ASSISTANT ==========
(function initChatbot() {
    const chatButton = document.getElementById('chatButton'), chatWindow = document.getElementById('chatWindow'), chatCloseBtn = document.getElementById('chatCloseBtn'), chatInput = document.getElementById('chatInput'), chatSendBtn = document.getElementById('chatSendBtn'), chatMessages = document.getElementById('chatMessages');
    if (!chatButton) return;
    chatButton.addEventListener('click', () => chatWindow.classList.toggle('open'));
    chatCloseBtn.addEventListener('click', () => chatWindow.classList.remove('open'));
    function addMessage(text, sender) {
        const msgDiv = document.createElement('div'); msgDiv.className = `chat-message ${sender}`;
        msgDiv.innerText = text; chatMessages.appendChild(msgDiv); chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    function showTyping() { const ind = document.createElement('div'); ind.className = 'chat-message assistant'; ind.id = 'typingIndicator'; ind.innerText = 'Typing...'; chatMessages.appendChild(ind); }
    function removeTyping() { const ind = document.getElementById('typingIndicator'); if (ind) ind.remove(); }
    function getBotReply(q) {
        const lower = q.toLowerCase();
        if (lower.includes('hello') || lower.includes('hi')) return "Hello! How can I help with breast cancer analysis?";
        if (lower.includes('model') && lower.includes('work')) return "The system uses EfficientNetB4 deep learning CNN trained on histopathology images to differentiate benign from malignant patterns.";
        if (lower.includes('accuracy')) return "The model achieves about 92-95% accuracy on validation data.";
        if (lower.includes('benign') || lower.includes('malignant')) return "Benign = non-cancerous. Malignant = cancerous. Our AI helps classify these.";
        if (lower.includes('training') || lower.includes('train')) return "Go to the Training tab, upload a ZIP with 'benign' and 'malignant' folders, set epochs, and click Start Training.";
        if (lower.includes('batch')) return "Batch mode lets you select a whole folder. The system processes them in chunks of 50 and gives a summary table + CSV download.";
        if (lower.includes('report') || lower.includes('recommendation')) return "After each prediction you get a detailed pathology report and care recommendations. You can also save a PDF via the 'Save Report as PDF' button.";
        if (lower.includes('dataset')) return "You can use BreakHis public dataset or your own labelled images. ZIP must contain two folders exactly named 'benign' and 'malignant'.";
        return "I'm not sure about that. Please ask about: model, accuracy, benign vs malignant, training, batch mode, reports, dataset, or help.";
    }
    async function handleUserMessage() {
        const msg = chatInput.value.trim(); if (!msg) return;
        addMessage(msg, 'user'); chatInput.value = '';
        showTyping();
        setTimeout(() => { const reply = getBotReply(msg); removeTyping(); addMessage(reply, 'assistant'); }, 500);
    }
    chatSendBtn.addEventListener('click', handleUserMessage);
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleUserMessage(); });
})();

// ========== BROWSER NAVIGATION ==========
document.addEventListener('DOMContentLoaded', function() {
    const backBtn = document.getElementById('backBtn'), forwardBtn = document.getElementById('forwardBtn'), refreshBtn = document.getElementById('refreshBtn'), stopBtn = document.getElementById('stopBtn'), homeBtn = document.getElementById('homeBtn'), addressBar = document.getElementById('addressBar'), goBtn = document.getElementById('goBtn');
    if (!backBtn) return;
    backBtn.addEventListener('click', (e) => { e.preventDefault(); window.history.back(); });
    forwardBtn.addEventListener('click', (e) => { e.preventDefault(); window.history.forward(); });
    refreshBtn.addEventListener('click', (e) => { e.preventDefault(); location.reload(); });
    stopBtn.addEventListener('click', (e) => { e.preventDefault(); window.stop(); });
    homeBtn.addEventListener('click', (e) => { e.preventDefault(); window.location.href = '/'; });
    function goToUrl() { let url = addressBar.value.trim(); if (!url) return; if (!url.startsWith('http://') && !url.startsWith('https://')) url = 'https://' + url; window.open(url, '_blank'); }
    goBtn.addEventListener('click', goToUrl);
    addressBar.addEventListener('keypress', (e) => { if (e.key === 'Enter') { e.preventDefault(); goToUrl(); } });
    function updateAddressBar() { addressBar.value = window.location.href; }
    updateAddressBar();
    window.addEventListener('popstate', updateAddressBar);
});