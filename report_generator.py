# utils/report_generator.py
def generate_report(prediction, confidence):
    if prediction == "Malignant":
        report = f"""
        <h4>Pathology Report Summary</h4>
        <p><strong>Finding:</strong> Malignant tumor detected with {confidence*100:.1f}% confidence.</p>
        <p><strong>Characteristics:</strong> The histopathological analysis indicates abnormal cell proliferation, nuclear pleomorphism, and possible invasion of surrounding tissue structures.</p>
        <p><strong>Clinical Significance:</strong> Immediate follow-up and confirmatory tests are strongly advised.</p>
        <p><strong>Suggested Next Steps:</strong></p>
        <ul>
            <li>Consult with an oncologist within 1 week</li>
            <li>Consider core needle biopsy for confirmation</li>
            <li>Evaluate hormone receptor status (ER/PR/HER2)</li>
            <li>Discuss treatment options: surgery, chemotherapy, radiation, or targeted therapy</li>
        </ul>
        """
    else:
        report = f"""
        <h4>Pathology Report Summary</h4>
        <p><strong>Finding:</strong> Benign findings with {confidence*100:.1f}% confidence.</p>
        <p><strong>Characteristics:</strong> The tissue architecture appears mostly normal with no significant atypia or malignant transformation observed.</p>
        <p><strong>Clinical Significance:</strong> Low risk of breast cancer based on this sample. Routine monitoring recommended.</p>
        <p><strong>Suggested Next Steps:</strong></p>
        <ul>
            <li>Regular clinical breast exams every 6-12 months</li>
            <li>Continue routine mammography as per age guidelines</li>
            <li>Breast self-awareness education</li>
            <li>Follow-up imaging if any changes occur</li>
        </ul>
        """
    return report

def generate_recommendations(prediction, confidence):
    if prediction == "Malignant":
        recommendations = f"""
        <div class="rec-card urgent">
            <h5>⚠️ Urgent Recommendations</h5>
            <ul>
                <li><strong>Immediate steps:</strong> Schedule an appointment with a breast surgeon or oncologist within 1-2 weeks.</li>
                <li><strong>Diagnostic confirmation:</strong> Request a confirmatory biopsy if not already performed.</li>
                <li><strong>Imaging:</strong> Consider breast MRI or ultrasound for staging.</li>
                <li><strong>Genetic counseling:</strong> Discuss BRCA testing if family history present.</li>
                <li><strong>Support:</strong> Reach out to patient support groups and social workers.</li>
            </ul>
        </div>
        """
    else:
        recommendations = f"""
        <div class="rec-card routine">
            <h5>✅ Routine Recommendations</h5>
            <ul>
                <li><strong>Monitoring:</strong> Continue routine breast health surveillance.</li>
                <li><strong>Lifestyle:</strong> Maintain healthy weight, exercise regularly, limit alcohol.</li>
                <li><strong>Follow-up:</strong> Repeat imaging in 6-12 months or as clinically indicated.</li>
                <li><strong>Education:</strong> Learn breast self-examination techniques.</li>
                <li><strong>Risk reduction:</strong> Discuss risk factors with your primary care provider.</li>
            </ul>
        </div>
        """
    return recommendations