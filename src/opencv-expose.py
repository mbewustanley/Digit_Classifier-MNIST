import cv2
import keras
import numpy as np




# 1. LOAD YOUR TRAINED MODEL HERE
# ==========================================
# Example for Keras/TensorFlow:
# from tensorflow.keras.models import load_model
# model = load_model('your_digit_model.h5')

# load model
#to later load model, use keras.saving.load_model('artifacts/DeepML/cnn_model.keras')
cnn_model = keras.saving.load_model('artifacts/DeepML/cnn_model.keras')
print(cnn_model.summary())



# 2. CONFIGURATION & CAMERA SETUP
# ==========================================
# Target dimensions your model expects (e.g., MNIST uses 28x28)
TARGET_W, TARGET_H = 28, 28 

# Open default webcam (0 is usually the built-in or first USB camera)
cap = cv2.VideoCapture(0)

# Check if webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Pipeline started. Position a written digit inside the green box.")
print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Get frame dimensions to place a bounding box in the center
    frame_h, frame_w, _ = frame.shape
    box_size = 200
    x1 = int((frame_w - box_size) / 2)
    y1 = int((frame_h - box_size) / 2)
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw the Region of Interest (ROI) bounding box on the display frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)



    # 3. EXTRACTION & PREPROCESSING (The Pipeline)
    # ==========================================
    # Crop the image to just the area inside our box
    roi = frame[y1:y2, x1:x2]

    # Step A: Convert to Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Step B: Clean image noise using Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step C: Thresholding (Ensures high contrast black/white separation)
    # Adjust cv2.THRESH_BINARY vs cv2.THRESH_BINARY_INV depending on if your 
    # model was trained on white text with black backgrounds or vice versa.
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Step D: Resize to model's expected shape (e.g., 28x28)
    resized = cv2.resize(thresh, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    # Step E: Normalize pixel intensities from [0-255] to [0.0 - 1.0]
    normalized = resized.astype("float32") / 255.0

    # Step F: Add batch and channel dimensions (Shape becomes: 1, 28, 28, 1)
    # Adjust this layout if your model expects channel-first format (1, 1, 28, 28)
    input_tensor = np.expand_dims(normalized, axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=-1)



    
    # 4. MODEL INFERENCE & VISUALIZATION
    # ==========================================
    # Run the preprocessed array through your classifier
    predictions = cnn_model.predict(input_tensor)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Only show prediction if confidence passes a reasonable threshold
    if confidence > 0.70:
        display_text = f"Digit: {predicted_class} ({confidence*100:.1f}%)"
        
        # Overlay text onto the video stream
        cv2.putText(
            frame, 
            display_text, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )

    # Display windows (Live feedback + what the model actually sees)
    cv2.imshow("Live Stream (Digit Classifier)", frame)
    cv2.imshow("Model Input View (Preprocessed ROI)", thresh)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
