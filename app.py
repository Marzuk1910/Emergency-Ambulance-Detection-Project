import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
import google.generativeai as genai

# ==============================
# 🔐 Configure Gemini
# ==============================
genai.configure(api_key="AIzaSyA8AtkgkGDTmylA5bPZOGo15AkzP9QyleQ")
model_ai = genai.GenerativeModel("gemini-2.5-flash")

# ==============================
# 📦 Load CNN Model
# ==============================
model = load_model("ambulance_model.h5")

# ==============================
# 🚑 Main Function
# ==============================
def detect_ambulance(image):

    img = cv2.resize(image, (150, 150))
    img = img / 255.0
    img = np.reshape(img, (1, 150, 150, 3))

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        result = "Ambulance 🚑 Detected"

        prompt = """
        Ambulance detected at traffic signal.
        Traffic density: Medium.
        Give short smart traffic control instruction (3 lines only).
        """

        response = model_ai.generate_content(prompt)
        ai_decision = response.text

    else:
        result = "No Ambulance Detected"
        ai_decision = "Normal traffic signal operation."

    return result, ai_decision


# ==============================
# 🌐 Gradio Interface
# ==============================
interface = gr.Interface(
    fn=detect_ambulance,
    inputs=gr.Image(type="numpy"),
    outputs=["text", "text"],
    title="🚦 Smart Emergency Traffic System",
    description="Upload image to detect ambulance and generate AI-based traffic decision."
)

interface.launch()