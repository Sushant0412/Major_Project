# import os
# import io
# import base64
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import shap
# import matplotlib.pyplot as plt
# from flask import Flask, request, render_template, redirect, url_for, jsonify
# from dotenv import load_dotenv
# import markdown2
# import google.generativeai as genai  # Still used for the initial analysis

# # Load environment variables from .env file
# load_dotenv()

# app = Flask(__name__)

# # --- CONFIGURATION ---
# MODEL_PATH = "pneumonia_detector.keras"
# IMAGE_SIZE = (260, 260)

# # --- LOAD MODELS AND INITIALIZE SHAP ---
# try:
#     # Load Keras model
#     model = tf.keras.models.load_model(MODEL_PATH)
#     print(f"✅ Keras model loaded successfully from {MODEL_PATH}")

#     # Prepare a background dataset for SHAP
#     # We take a small sample from the training data for the explainer.
#     # NOTE: This assumes you have a 'train' subdirectory in your DATA_DIR.
#     background_data_dir = "chest_xray_data/"
#     if os.path.exists(background_data_dir):
#         background_dataset = tf.keras.utils.image_dataset_from_directory(
#             background_data_dir,
#             seed=123,
#             image_size=IMAGE_SIZE,
#             batch_size=16,  # Load a small batch
#         )
#         background_images = next(iter(background_dataset))[0]
#         # Initialize SHAP GradientExplainer
#         explainer = shap.GradientExplainer(model, background_images)
#         print("SHAP explainer initialized successfully.")
#     else:
#         explainer = None
#         print(
#             "Warning: SHAP background data directory not found. Explanations will not be available."
#         )

# except Exception as e:
#     print(f"❌ Error during model loading or SHAP initialization: {e}")
#     model = None
#     explainer = None


# def preprocess_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize(IMAGE_SIZE)
#     img_array = tf.keras.utils.img_to_array(img)
#     img_batch = np.expand_dims(img_array, axis=0)
#     return img, img_batch


# def get_report(pil_image, keras_prediction):
#     # This function remains to provide the initial text-based AI analysis
#     if not shap:
#         return "Shap model is not available."
#     prompt_parts = [
#         "You are an expert radiologist specializing in the interpretation of chest X-rays for diagnosing pneumonia. Your task is to analyze the provided chest X-ray image.",
#         f"The initial deep learning model predicted this case as: **{keras_prediction}**.",
#         "Please provide a detailed summary of your findings in a structured report. This report should be useful for a consulting physician.",
#         "Your analysis must:",
#         "1. Confirm or contest the initial model's prediction.",
#         "2. Describe any visible radiological signs (e.g., opacities, consolidations, infiltrates, air bronchograms, pleural effusions). If none are present, state that clearly.",
#         "3. Clearly reference specific areas of the lung (e.g., 'in the right lower lobe', 'apical region').",
#         "4. Validate your statements by describing what you see in the image (or the lack of pathological findings).",
#         "5. Conclude with a final impression and recommendation.",
#         "\n**Radiology Report**\n---",
#         pil_image,
#     ]
#     try:
#         response = shap.generate_content(prompt_parts)
#         return response.text
#     except Exception as e:
#         return f"Could not generate Gemini report. Error: {e}"


# @app.route("/", methods=["GET"])
# def index():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return redirect(request.url)
#     file = request.files["file"]
#     if not file or file.filename == "":
#         return redirect(request.url)

#     image_bytes = file.read()
#     if model is None:
#         return "Server Error: Keras model not loaded.", 500

#     pil_image, img_batch = preprocess_image(image_bytes)
#     prediction = model.predict(img_batch)
#     score = prediction[0][0]
#     keras_diagnosis = "Pneumonia" if score > 0.5 else "Normal"

#     gemini_report_text = get_report(pil_image, keras_diagnosis)
#     gemini_analysis_html = markdown2.markdown(gemini_report_text)
#     encoded_img = base64.b64encode(image_bytes).decode("utf-8")

#     return render_template(
#         "results.html",
#         keras_result=keras_diagnosis,
#         gemini_analysis=gemini_analysis_html,
#         uploaded_image=encoded_img,
#     )


# # --- NEW SHAP EXPLANATION ENDPOINT ---
# @app.route("/explain", methods=["POST"])
# def explain_prediction():
#     if explainer is None:
#         return jsonify({"error": "SHAP explainer is not available."}), 500

#     data = request.get_json()
#     if not data or "image" not in data:
#         return jsonify({"error": "Missing image data"}), 400

#     # Decode and preprocess the image
#     image_bytes = base64.b64decode(data["image"])
#     _, img_batch = preprocess_image(image_bytes)

#     # Calculate SHAP values
#     shap_values = explainer.shap_values(img_batch)

#     # --- Generate and save the SHAP plot ---
#     # We create a plot in memory without displaying it on the server
#     fig, ax = plt.subplots()
#     shap.image_plot(shap_values, -img_batch, ax=ax, show=False)
#     fig.tight_layout()

#     # Save the plot to an in-memory buffer
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight")
#     plt.close(fig)
#     buf.seek(0)

#     # Encode the image to base64 to send to the frontend
#     explanation_img_base64 = base64.b64encode(buf.read()).decode("utf-8")

#     return jsonify(
#         {"explanation_image": f"data:image/png;base64,{explanation_img_base64}"}
#     )


# if __name__ == "__main__":
#     app.run(debug=True)


import os
import random  # For generating random numbers
from datetime import datetime  # For formatting the date
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
import shap
import matplotlib

matplotlib.use(
    "Agg"
)  # Use a non-interactive backend for Matplotlib to prevent server crashes
import matplotlib.pyplot as plt
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    jsonify,
    send_from_directory,
)
from dotenv import load_dotenv
import markdown2
import google.generativeai as genai
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "pneumonia_detector.keras"
IMAGE_SIZE = (260, 260)
API_KEY = os.getenv("GEMINI_API_KEY")

# --- LOAD MODELS AND INITIALIZE SHAP ---
try:
    # Load Keras model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Keras model loaded successfully from {MODEL_PATH}")

    # Prepare a background dataset for SHAP
    # This path needs to point to your local training data directory
    background_data_dir = "chest_xray_data/train"
    if os.path.exists(background_data_dir):
        background_dataset = tf.keras.utils.image_dataset_from_directory(
            background_data_dir, seed=123, image_size=IMAGE_SIZE, batch_size=16
        )
        background_images = next(iter(background_dataset))[0]
        explainer = shap.GradientExplainer(model, background_images)
        # print("✅ SHAP explainer initialized successfully.")
    else:
        explainer = None
        # print(
        #     f"⚠️ Warning: SHAP background data directory not found at '{background_data_dir}'. Explanations will not be available."
        # )

except Exception as e:
    print(f"❌ Error during model loading or SHAP initialization: {e}")
    model = None
    explainer = None

# --- CONFIGURE GEMINI API ---
try:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    gemini_model = None


def preprocess_image(image_bytes):
    """Preprocesses raw image bytes for the Keras model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_batch


def get_gemini_report(pil_image, keras_prediction):
    """Generates a detailed report from Gemini, including image validation."""
    if not gemini_model:
        return "Gemini model is not available."
    prompt_parts = [
        "First, analyze the provided image to determine if it is a medical radiograph of a human chest (a chest X-ray).",
        "IF THE IMAGE IS NOT a chest X-ray, your ONLY response should be the following exact text: 'Image Invalid: The uploaded image does not appear to be a chest X-ray. Please upload a relevant medical image for analysis.'",
        "IF AND ONLY IF the image IS a chest X-ray, then proceed with the following instructions:",
        "---",
        "You are an expert radiologist specializing in the interpretation of chest X-rays for diagnosing pneumonia. Your task is to analyze the provided chest X-ray image. Give one line space after each point",
        f"The initial deep learning model predicted this case as: **{keras_prediction}**.",
        "Please provide a detailed summary of your findings in a structured report. This report should be useful for a consulting physician.",
        "Your analysis must:",
        "1. Confirm or contest the initial model's prediction.\n",
        "2. Describe any visible radiological signs (e.g., opacities, consolidations, infiltrates, air bronchograms, pleural effusions). If none are present, state that clearly.\n",
        "3. Clearly reference specific areas of the lung (e.g., 'in the right lower lobe', 'apical region').\n",
        "4. Validate your statements by describing what you see in the image (or the lack of pathological findings).\n",
        "5. Conclude with a final impression and recommendation.\n",
        "\n**Radiology Report**\n---",
        pil_image,
    ]
    try:
        response = gemini_model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"Could not generate Gemini report. Error: {e}"


def generate_formatted_report_html(
    initial_report_html, patient_name, study_date, patient_sex, ipd_no, dept_no
):
    """Calls Gemini to format the analysis into a professional HTML report."""
    if not gemini_model:
        return "Gemini model is not available."

    soup = BeautifulSoup(initial_report_html, "html.parser")
    initial_report_text = soup.get_text()

    # --- NEW: Format the date ---
    try:
        # Convert 'YYYY-MM-DD' from form to 'DD-Mon-YYYY' for report
        date_obj = datetime.strptime(study_date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d-%b-%Y").upper()
    except (ValueError, TypeError):
        formatted_date = "Not Provided"

    # This is the detailed HTML template prompt for Gemini
    html_template_prompt = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Radiology Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f4f4f4; color: #333; }
        .report-container { max-width: 800px; margin: auto; background: #fff; padding: 30px; border: 1px solid #ddd; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .report-header { width: 100%; border-collapse: collapse; margin-bottom: 25px; border: 2px solid #333; }
        .report-header td { border: 1px solid #333; padding: 8px; font-size: 14px; }
        .report-header td:nth-child(odd) { font-weight: bold; width: 15%; }
        .report-header td:nth-child(even) { width: 35%; }
        .report-title { text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 20px; text-decoration: underline; }
        .report-body p { margin: 12px 0; font-size: 16px; }
        .report-body ul { padding-left: 20px; }
        .report-body li { margin-bottom: 8px; }
        .signature-section { margin-top: 60px; text-align: right; }
        .signature-image { width: 180px; margin-bottom: 5px; }
        .signature-section p { margin: 0; line-height: 1.4; font-size: 15px; }
    </style>
</head>
<body>
    <div class="report-container">
        <table class="report-header">
            <tr>
                <td>Patient Name:</td> <td>[Patient Name]</td>
                <td>Study Date:</td> <td>[DD-Mon-YYYY]</td>
            </tr>
            <tr>
                <td>Referring Doctor:</td> <td>Dr. PM Purohit</td>
                <td>Age:</td> <td>[Age]</td>
            </tr>
            <tr>
                <td>Sex:</td> <td>[Sex]</td>
                <td>UHID:</td> <td>[UHID]</td>
            </tr>
            <tr>
                <td>IPD No.:</td> <td>[IPD No]</td>
                <td>Dept. No.:</td> <td>[Dept No]</td>
            </tr>
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
            <svg class="signature-image" viewBox="0 0 300 100" xmlns="http://www.w3.org/2000/svg"><path d="M 10,70 C 20,20 60,20 80,70 C 100,120 120,20 150,50 C 180,80 200,20 230,60 C 260,100 280,40 290,70" stroke="black" fill="transparent" stroke-width="2"/></svg>
            <p><strong>Dr.P.M. Purohit MD,DMRE</strong></p>
            <p>Consultant Radiologist</p>
        </div>
    </div>
</body>
</html>"""

    # --- NEW: Pre-populate the template with our data ---
    populated_template = html_template_prompt.replace("[Patient Name]", patient_name)
    populated_template = populated_template.replace("[DD-Mon-YYYY]", formatted_date)
    populated_template = populated_template.replace("[Sex]", patient_sex)
    populated_template = populated_template.replace("[IPD No]", ipd_no)
    populated_template = populated_template.replace("[Dept No]", dept_no)

    prompt = [
        "You are a professional medical report formatter.",
        "Reformat the given X-ray analysis text into the provided HTML template.",
        "The template has most patient data already filled in.",
        "Your job is to parse the analysis text and use it to fill in the REMAINING placeholders, such as [Finding 1], [Finding 2], [Age], [UHID], and the statements about the heart, aorta, etc.",
        "Use sensible defaults like 'Not assessed' or 'Normal' if the analysis text doesn't mention a specific point.",
        "Your entire response must be ONLY the raw HTML code, starting with <!DOCTYPE html> and ending with </html>. Do not include markdown fences (```html) or any other text.",
        "\n--- HTML TEMPLATE START ---",
        populated_template,  # Use the pre-populated template
        "\n--- HTML TEMPLATE END ---",
        "\nHere is the AI’s Initial Analysis text to reformat:",
        initial_report_text,
    ]
    try:
        response = gemini_model.generate_content(prompt)
        html_content = response.text.strip()
        return html_content
    except Exception as e:
        print(f"❌ Could not generate formatted report. Error: {e}")
        return f"<html><body><h1>Error</h1><p>Could not generate formatted report: {e}</p></body></html>"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if not file or file.filename == "":
        return redirect(request.url)

    # --- NEW: Get form data ---
    patient_name = request.form.get("name", "Not Provided")
    study_date = request.form.get("date")  # Gets date as 'YYYY-MM-DD'
    patient_sex = request.form.get("sex", "Not Provided")

    # --- NEW: Generate random hardcoded numbers ---
    ipd_no = random.randint(10000, 99999)
    dept_no = random.randint(1, 15)  # Example: random dept number 1-15

    image_bytes = file.read()
    if model is None:
        return "Server Error: Keras model not loaded.", 500

    pil_image, img_batch = preprocess_image(image_bytes)
    prediction = model.predict(img_batch)
    score = prediction[0][0]
    keras_diagnosis = "Pneumonia" if score > 0.5 else "Normal"
    gemini_report_text = get_gemini_report(pil_image, keras_diagnosis)

    is_valid_image = not gemini_report_text.strip().startswith("Image Invalid:")

    if is_valid_image:
        gemini_analysis_html = markdown2.markdown(gemini_report_text)
    else:
        keras_diagnosis = "Invalid Image"
        gemini_analysis_html = (
            f'<p style="color: #f44336; font-weight: bold;">{gemini_report_text}</p>'
        )

    encoded_img = base64.b64encode(image_bytes).decode("utf-8")

    # --- UPDATED: Pass all data to results.html ---
    return render_template(
        "results.html",
        keras_result=keras_diagnosis,
        gemini_analysis=gemini_analysis_html,
        uploaded_image=encoded_img,
        is_valid_image=is_valid_image,
        # Pass patient/report data to the next page
        patient_name=patient_name,
        study_date=study_date,
        patient_sex=patient_sex,
        ipd_no=ipd_no,
        dept_no=dept_no,
    )


@app.route("/explain", methods=["POST"])
def explain_prediction():
    if explainer is None:
        return jsonify({"error": "SHAP explainer is not available."}), 500
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image data"}), 400

    image_bytes = base64.b64decode(data["image"])
    _, img_batch = preprocess_image(image_bytes)

    shap_values = explainer.shap_values(img_batch)

    fig, ax = plt.subplots()
    shap.image_plot(shap_values, -img_batch, ax=ax, show=False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    explanation_img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return jsonify(
        {"explanation_image": f"data:image/png;base64,{explanation_img_base64}"}
    )


@app.route("/generate_report", methods=["POST"])
def generate_report_endpoint():
    """
    Handles the AJAX request to generate the final formatted report.
    This function now reads all patient data from the JSON payload.
    """
    try:
        data = request.get_json()

        # --- FIX: Extract all data from the JSON payload ---
        report_html = data.get("report_html", "")
        patient_name = data.get("patient_name", "Not Provided")
        study_date = data.get("study_date", "")
        patient_sex = data.get("patient_sex", "Not Provided")
        ipd_no = str(data.get("ipd_no", "N/A"))
        dept_no = str(data.get("dept_no", "N/A"))

        # --- FIX: Pass all 6 arguments to the formatter function ---
        # (Make sure your generate_formatted_report_html function accepts these)
        report_html_content = generate_formatted_report_html(
            report_html, patient_name, study_date, patient_sex, ipd_no, dept_no
        )

        report_path = os.path.join(app.root_path, "templates", "report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html_content)

        return jsonify(
            {
                "success": True,
                "message": "Report generated successfully.",
                "report_url": url_for("view_report"),
                "download_url": url_for("download_report"),
            }
        )
    except Exception as e:
        # Added a print statement for better debugging
        print(f"❌ Error in /generate_report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/view_report")
def view_report():
    """Serves the generated report.html page."""
    return render_template("report.html")


@app.route("/download_report")
def download_report():
    """Provides the generated report.html file as a download."""
    return send_from_directory("templates", "report.html", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
