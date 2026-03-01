Emergency-Ambulance-Detection-Project
“This project uses CNN to classify ambulance vehicles from images. I trained the model using TensorFlow and deployed it using Gradio interface for real-time prediction.”
Emergency Ambulance Detection using CNN
Project Overview
This project detects Ambulance vs Non-Ambulance vehicles using Computer Vision and Deep Learning.
The system uses:
- CNN (Convolutional Neural Network)
- TensorFlow / Keras
- OpenCV for image preprocessing
- Gradio for web interface
# How It Works
1. Input image is uploaded.
2. Image is preprocessed using OpenCV.
3. CNN model predicts:
   - Ambulance
   - Non-Ambulance
4. Output is displayed in Gradio interface.
## Technologies Used
- Python
- TensorFlow
- OpenCV
- Gradio
## How to Run
```bash
pip install -r requirements.txt
python app.py
