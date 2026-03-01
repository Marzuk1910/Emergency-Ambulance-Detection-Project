import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("ambulance_model.h5")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for model
    img = cv2.resize(frame, (150, 150))
    img = img / 255.0
    img = np.reshape(img, (1, 150, 150, 3))

    # Predict
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        label = "Ambulance 🚑"
        color = (0, 255, 0)
    else:
        label = "No Ambulance"
        color = (0, 0, 255)

    # Show text on frame
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.imshow("Ambulance Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()