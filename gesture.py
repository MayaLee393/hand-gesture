# https://www.geeksforgeeks.org/face-and-hand-landmarks-detection-using-python-mediapipe-opencv/

# Import Libraries
import cv2
import time
import mediapipe as mp
from collections import deque
import pyautogui

# Track recent gestures
gesture_buffer = deque(maxlen=5)  # store last 5 gestures
last_triggered = None

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# Finger indices
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
FINGER_PIPS = [3, 6, 10, 14, 18]

# Detect which fingers are up
def get_finger_status(landmarks, is_left):
    fingers = []

    # Thumb - check x-axis
    if is_left:
        fingers.append(landmarks[4].x > landmarks[3].x)
    else:
        fingers.append(landmarks[4].x < landmarks[3].x)

    # Other fingers - check y-axis
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers  # [thumb, index, middle, ring, pinky]

def classify_gesture(status, landmarks):
    thumb_tip = landmarks[4]
    wrist = landmarks[0]
    index_mcp = landmarks[5]

    vertical_thumb = thumb_tip.y - wrist.y
    horizontal_thumb = abs(thumb_tip.x - wrist.x)

    # Thumbs up if thumb tip is clearly above wrist and other fingers down
    if status == [True, False, False, False, False] and vertical_thumb < -0.1:
        return "Thumbs Up"
    # Thumbs down if thumb tip is clearly below wrist and other fingers down
    elif status == [True, False, False, False, False] and vertical_thumb > 0.1:
        return "Thumbs Down"
    elif status == [False, False, False, False, False]:
        return "Fist"
    elif status == [True, True, True, False, False]:
        return "Peace"
    elif status == [False, True, True, True, True]:
        return "Wave"
    else:
        return "Unknown"


# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()

    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    gesture = "No Hand"

    # Check for right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        landmarks = results.right_hand_landmarks.landmark
        status = get_finger_status(landmarks, is_left=False)
        gesture = classify_gesture(status, landmarks)
        # Display finger status on screen
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        status_labels = [f"{finger}: {'Up' if s else 'Down'}" for finger, s in zip(finger_names, status)]
        status_text = ", ".join(status_labels)

        cv2.putText(image, status_text, (10, 160), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)
        # Display landmark positions
        h, w, _ = frame.shape
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(image, f"{i}:{cx},{cy}", (cx, cy - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)

    # Check for left hand
    elif results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        landmarks = results.left_hand_landmarks.landmark
        status = get_finger_status(landmarks, is_left=True)
        gesture = classify_gesture(status, landmarks)
        # Display finger status on screen
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        status_labels = [f"{finger}: {'Up' if s else 'Down'}" for finger, s in zip(finger_names, status)]
        status_text = ", ".join(status_labels)

        cv2.putText(image, status_text, (10, 160), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)
        # Display landmark positions
        h, w, _ = frame.shape
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(image, f"{i}:{cx},{cy}", (cx, cy - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)

    # Trigger actions based on gesture
    # Add current gesture to buffer
    gesture_buffer.append(gesture)

    # Check if the buffer is stable
    if len(gesture_buffer) == gesture_buffer.maxlen:
        if all(g == gesture_buffer[0] for g in gesture_buffer):
            stable_gesture = gesture_buffer[0]

            # Trigger action only if gesture is different from last triggered
            if stable_gesture != last_triggered:
                if stable_gesture == "Fist":
                    print("Stop")
                elif stable_gesture == "Thumbs Up":
                    print("Approved")
                elif stable_gesture == "Thumbs Down":
                    print("Rejected")
                elif stable_gesture == "Peace":
                    print("Peace")
                    screen_width, screen_height = pyautogui.size()
                    center_x = screen_width // 2
                    center_y = screen_height // 2
                    pyautogui.moveTo(center_x, center_y)
                    pyautogui.doubleClick()
                elif stable_gesture == "Wave":
                    print("Hello")
                    pyautogui.scroll(-200)

                last_triggered = stable_gesture


    # Draw gesture label and FPS
    cv2.putText(image, gesture, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
      image,
      results.face_landmarks,
      mp_holistic.FACEMESH_CONTOURS,
      mp_drawing.DrawingSpec(
        color=(255,0,255),
        thickness=1,
        circle_radius=1
      ),
      mp_drawing.DrawingSpec(
        color=(0,255,255),
        thickness=1,
        circle_radius=1
      )
    )

    # Drawing Right hand Land Marks
    mp_drawing.draw_landmarks(
      image, 
      results.right_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )

    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
      image, 
      results.left_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
    
    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
    
    # Displaying FPS on the image
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()

# Code to access landmarks
for landmark in mp_holistic.HandLandmark:
    print(landmark, landmark.value)

print(mp_holistic.HandLandmark.WRIST.value)