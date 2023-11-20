import dlib
import cv2
import numpy as np

# Load the pre-trained facial landmark detection model
predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Path to the pre-trained model file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load the input image
image_path = 'target.jpg'  # Path to the input image file
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Iterate over detected faces
for face in faces:
    # Predict facial landmarks
    landmarks = predictor(gray, face)

    # Convert landmarks to numpy array
    landmarks_array = np.zeros((68, 2), dtype=int)  # Modified line

    for i in range(0, 68):
        landmarks_array[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # Save landmarks to a text file
    np.savetxt('target.txt', landmarks_array, fmt='%d')

    # Draw facial landmarks on the image
    for (x, y) in landmarks_array:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

# Save the output image
cv2.imwrite('target_landmarks.png', image)