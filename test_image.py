import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import google.generativeai as genai

# ==============================
# 🔐 CONFIGURE GEMINI API
# ==============================
genai.configure(api_key="AIzaSyA8AtkgkGDTmylA5bPZOGo15AkzP9QyleQ")
model_ai = genai.GenerativeModel("gemini-2.5-flash")

# ==============================
# 📦 LOAD TRAINED CNN MODEL
# ==============================
model = load_model("ambulance_model.h5")

# ==============================
# 🖼 LOAD IMAGE
# ==============================
image_path = "Ambulance.jpg"
img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Image not found. Check file name.")
    exit()

# Preprocess image
img_resized = cv2.resize(img, (150, 150))
img_normalized = img_resized / 255.0
img_reshaped = np.reshape(img_normalized, (1, 150, 150, 3))

# ==============================
# 🔍 PREDICT
# ==============================
prediction = model.predict(img_reshaped)

if prediction[0][0] > 0.5:
    result = "Ambulance 🚑"
    print("Prediction:", result)

    print("DEBUG: Entered Gemini block ✅")

    traffic_density = "Medium"

    prompt = f"""
    Ambulance detected at traffic signal.
    Traffic density: {traffic_density}.
    Suggest intelligent traffic signal control strategy.
    """

    print("DEBUG: Calling Gemini...")

try:
    response = model_ai.generate_content(prompt)
    print("DEBUG: Gemini responded ✅")

    print("\n🚦 AI Traffic Decision:\n")
    print(response.text)

except Exception as e:
    print("❌ Gemini Error:", e)

else:
    result = "No Ambulance"
    print("Prediction:", result)

# ==============================
# 🖥 DISPLAY IMAGE
# ==============================
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(result)
plt.axis("off")
plt.show()