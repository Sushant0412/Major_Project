import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
import google.generativeai as genai
from flask import Flask, request, render_template, redirect, url_for, jsonify
from dotenv import load_dotenv
import markdown2  # <--- 1. IMPORT THE NEW LIBRARY
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "pneumonia_detector.keras"
IMAGE_SIZE = (260, 260)
API_KEY = os.getenv("GEMINI_API_KEY")

# --- LOAD KERAS MODEL ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Keras model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading Keras model: {e}")
    model = None

# --- CONFIGURE GEMINI API ---
try:
    genai.configure(api_key=API_KEY)
    # Use 'gemini-pro-vision' for multimodal tasks
    gemini_model = genai.GenerativeModel(model_name="gemini-2.5-pro")
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    gemini_model = None


def preprocess_image(image_bytes):
    """
    Preprocesses the image to be compatible with the Keras model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_batch  # Return both PIL image and batch


def get_gemini_report(pil_image, keras_prediction):
    """
    Generates a detailed report from the Gemini API.
    """
    if not gemini_model:
        return "Gemini model is not available."

    prompt_parts = [
        "You are an expert radiologist specializing in the interpretation of chest X-rays for diagnosing pneumonia. Your task is to analyze the provided chest X-ray image.",
        f"The initial model predicted this case as: **{keras_prediction}**.",
        "Please provide a detailed summary of your findings in a structured report, useful for a consulting physician.",
        "Your analysis must:",
        "1. Confirm or contest the initial model's prediction.",
        "2. Describe any visible radiological signs (e.g., opacities, consolidations). If none are present, state that clearly.",
        "3. Clearly reference specific areas of the lung (e.g., 'in the right lower lobe').",
        "4. Conclude with a final impression and recommendation.",
        "\n**Radiology Report**\n---",
        pil_image,
    ]
    try:
        response = gemini_model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"Could not generate Gemini report. Error: {e}"


@app.route("/", methods=["GET"])
def index():
    """Renders the main upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handles the image upload and prediction."""
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "" or not file:
        return redirect(request.url)

    image_bytes = file.read()

    if model is None:
        return "Server Error: Keras model not loaded.", 500

    pil_image, img_batch = preprocess_image(image_bytes)
    prediction = model.predict(img_batch)
    score = prediction[0][0]

    keras_diagnosis = "Pneumonia" if score > 0.5 else "Normal"

    # --- Gemini API Analysis ---
    gemini_report_text = get_gemini_report(pil_image, keras_diagnosis)

    # <--- 2. CONVERT MARKDOWN TO HTML ---
    # This turns the text report into formatted HTML
    gemini_analysis_html = markdown2.markdown(gemini_report_text)

    encoded_img = base64.b64encode(image_bytes).decode("utf-8")

    return render_template(
        "results.html",
        keras_result=keras_diagnosis,
        gemini_analysis=gemini_analysis_html,  # Pass the HTML version
        uploaded_image=encoded_img,
    )


def generate_formatted_report(pil_image, initial_report_html):
    """
    Calls Gemini to generate a hospital-style HTML radiology report.
    """
    if not gemini_model:
        return "Gemini model is not available."

    # Clean any prior HTML
    soup = BeautifulSoup(initial_report_html, "html.parser")
    initial_report_text = soup.get_text()

    # Structured hospital report prompt
    prompt = [
        "You are a professional medical report formatter.",
        "Reformat the given X-ray analysis into the provided HTML template exactly — keep the structure, style, and placeholders as shown below.",
        "Replace the placeholder text (findings, patient name, UHID, etc.) appropriately using information derived from the analysis and general defaults if missing.",
        "Return only valid HTML (do not include explanations or extra text).",
        "\n--- TEMPLATE START ---",
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radiology Report - NSSH.1215787</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            background-color: #f4f4f4;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        .report-container {
            max-width: 800px;
            margin: auto;
            background: #fff;
            padding: 30px;
            border: 1px solid #ddd;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .report-header {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
            border: 2px solid #333;
        }
        .report-header td {
            border: 1px solid #333;
            padding: 8px;
            font-size: 14px;
        }
        .report-header td:nth-child(odd) {
            font-weight: bold;
            width: 15%;
        }
        .report-header td:nth-child(even) {
            width: 35%;
        }
        .report-title {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
            text-decoration: underline;
        }
        .report-body p {
            margin: 10px 0;
            font-size: 16px;
        }
        .signature-section {
            margin-top: 60px;
            text-align: right;
        }
        .signature-image {
            width: 180px;
            margin-bottom: 5px;
        }
        .signature-section p {
            margin: 0;
            line-height: 1.4;
            font-size: 15px;
        }
    </style>
</head>
<body>

    <div class="report-container">
        <table class="report-header">
            <tbody>
                <tr>
                    <td>Patient Name:</td>
                    <td>[Patient Name]</td>
                    <td>Study Date:</td>
                    <td>[DD-Mon-YYYY]</td>
                </tr>
                <tr>
                    <td>Referring Doctor:</td>
                    <td>[Doctor]</td>
                    <td>Age:</td>
                    <td>[Age]</td>
                </tr>
                <tr>
                    <td>Sex:</td>
                    <td>[Sex]</td>
                    <td>UHID:</td>
                    <td>[UHID]</td>
                </tr>
                <tr>
                    <td>IPD No.:</td>
                    <td></td>
                    <td>Dept. No.:</td>
                    <td></td>
                </tr>
            </tbody>
        </table>

        <h2 class="report-title">CHEST</h2>

        <div class="report-body">
            <ul>
                <li>[Finding 1]</li>
            <li>[Finding 2]</li>
            <li>The heart size appears [normal/abnormal].</li>
            <li>The aortic knuckle is [normal/abnormal].</li>
            <li>Both costo-phrenic angles are [normal/abnormal].</li>
            <li>The bony thorax and both dome of diaphragms appear [normal/abnormal].</li>
            </ul>
        </div>

        <div class="signature-section">
            <svg class="signature-image" viewBox="0 0 300 100" xmlns="http://www.w3.org/2000/svg">
                <path d="M 10,70 C 20,20 60,20 80,70 C 100,120 120,20 150,50 C 180,80 200,20 230,60 C 260,100 280,40 290,70" stroke="black" fill="transparent" stroke-width="2"/>
            </svg>
            <p><strong>Dr.P.M. Purohit MD,DMRE</strong></p>
            <p>Consultant Radiologist</p>
        </div>
    </div>

</body>
</html>""",
        "\n--- TEMPLATE END ---",
        "\nHere is the provided chest X-ray image for reference:",
        pil_image,
        "\nHere is the AI’s Initial Analysis text to reformat:",
        initial_report_text,
    ]

    try:
        response = gemini_model.generate_content(prompt)
        print(response.text)
        return response.text  # Full HTML output
    except Exception as e:
        return f"Could not generate formatted report. Error: {e}"


# ------------------------------------
# Flask API endpoint for report output
# ------------------------------------
@app.route("/generate_report", methods=["POST"])
def generate_report_route():
    """
    Receives base64 image + report HTML, and returns a formatted HTML report.
    """
    try:
        data = request.get_json()
        if not data or "image" not in data or "report_html" not in data:
            return jsonify({"error": "Missing data"}), 400

        # Decode base64 image
        image_bytes = base64.b64decode(data["image"])
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Generate formatted HTML report
        formatted_report_html = generate_formatted_report(
            pil_image, data["report_html"]
        )

        return jsonify({"formatted_report": formatted_report_html})

    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
