import cv2 #for basic image processing
import numpy as np #array functions
import dlib #deep learning modules and face landmarks detection
import imutils
from imutils import face_utils #for basic conversion operations
from imutils.video import VideoStream
import time
import math

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks (1).dat')  # Adjust path as necessary

# Initialize the video stream (change src=1 to the appropriate camera index if needed)
vs = VideoStream(src=0).start()  # Use src=0 for default camera, src=1 for external camera, adjust as necessary
time.sleep(2.0)  # Allow camera to warm up

# Thresholds for drowsiness detection
ear_threshold = 0.25
blink_threshold = 3
mar_threshold = 0.79

# Initialize counter for blinking detection
COUNTER = 0

# Function to calculate distance between two points
def dist_btw_pts(x, y):
    return np.linalg.norm(x - y)

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    vert_left = dist_btw_pts(eye[1], eye[5])  # Vertical distance between eye landmarks
    vert_right = dist_btw_pts(eye[2], eye[4])
    horz = dist_btw_pts(eye[0], eye[3])  # Horizontal distance between eye landmarks
    ear = (vert_left + vert_right) / (2.0 * horz)  # Eye aspect ratio
    return ear

# Function to calculate mouth aspect ratio
def mouth_aspect_ratio(mouth):
    vert_left = dist_btw_pts(mouth[2], mouth[10])  # Vertical distance between mouth landmarks
    vert_right = dist_btw_pts(mouth[4], mouth[8])
    horz = dist_btw_pts(mouth[0], mouth[6])  # Horizontal distance between mouth landmarks
    mar = (vert_left + vert_right) / (2.0 * horz)  # Mouth aspect ratio
    return mar


predictions =[]

def count_ones_sets(predictions):
    count = 0
    in_one_sequence = False
    
    for i in range(len(predictions)):
        if predictions[i] == 1:
            if not in_one_sequence:
                # We just entered a sequence of 1's
                in_one_sequence = True
        elif predictions[i] == 0:
            if in_one_sequence:
                # We just exited a sequence of 1's
                count += 1
                in_one_sequence = False
                
    # Handle the case where the list ends with a sequence of 1's
    if in_one_sequence:
        count += 1
    
    return count


# Main loop for processing frames
while True:
    frame = vs.read()  # Read a frame from the video stream
    
    # Check if the frame was successfully captured
    if frame is None:
        print("Error: Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = detector(gray, 0)  # Detect faces in the grayscale frame


    drowsy = 0
    
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle around the face

        landmarks = predictor(gray, face)  # Predict facial landmarks
        landmarks = face_utils.shape_to_np(landmarks)  # Convert landmarks to numpy array

        # Calculate eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(landmarks[36:42])
        rightEAR = eye_aspect_ratio(landmarks[42:48])
        ear = (leftEAR + rightEAR) / 2.0  # Average eye aspect ratio

        # Check for drowsiness (blink detection)
        if ear < ear_threshold:
            COUNTER += 1
            if COUNTER >= blink_threshold:
                drowsy = 1
                cv2.putText(frame, "Sleepy!!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            COUNTER = 0

        # Calculate mouth aspect ratio
        mouth_pts = landmarks[49:69]
        mar = mouth_aspect_ratio(mouth_pts)

        # Check for yawning (mouth aspect ratio)
        if mar > mar_threshold:
            drowsy = 1
            cv2.putText(frame, "Yawning!!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        # Draw landmarks on the face
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    predictions.append(drowsy)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break



# Cleanup
cv2.destroyAllWindows()
vs.stop()  # Stop the video stream

# Print predictions
print("Predictions:", predictions)

