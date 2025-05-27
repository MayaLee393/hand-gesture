from collections import deque
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import pyautogui

width, height = pyautogui.size()
button_x, button_y = width/2, height/2

def handle_gesture(gesture):
    close = False
    match gesture:
        case "Open_Palm":
            pyautogui.scroll(-100)
        case "Thumb_Up":
            pyautogui.moveTo(button_x, button_y)
            if width/2 == button_x and height/2 == button_y:
                pyautogui.click(clicks=2)
            else:
                pyautogui.click()
        case "Pointing_Up":
            pyautogui.scroll(50)
        case "Thumb_Down":
            close = True
        case "Closed_Fist":
            pyautogui.click()
    return close

# getting the model from https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
model_path = 'gesture_recognizer.task'
BaseOptions = mp.tasks.BaseOptions
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

gesture = "Unknown"
gesture_buffer = deque(maxlen=5)
last_gesture = None
last_trigger = 0
cooldown = 1

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global gesture
    if result.gestures:
        gesture = str(result.gestures[0][0].category_name)
    else:
        gesture = "Unknown"
# creating the recognizer object
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_drawing = mp.solutions.drawing_utils
# start camera
capture = cv2.VideoCapture(0)
print("camera on")
while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (800, 500))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    recognizer.recognize_async(mp_image, frame_timestamp_ms)
    
    cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    gesture_buffer.append(gesture)
    close = False
    if len(gesture_buffer) == gesture_buffer.maxlen:
        stable = True
        for g in gesture_buffer:
            if g != gesture_buffer[0]:
                stable = False
        if stable and ( (gesture_buffer[0]!=last_gesture) or (time.time() - last_trigger > cooldown) ):
            close = handle_gesture(gesture_buffer[0])
            last_gesture = gesture_buffer[0]
            last_trigger = time.time()

    key = cv2.waitKey(5) & 0xFF
    if key == ord('s'):
        button_x, button_y = pyautogui.position()
    cv2.putText(frame, f"Saved cords: ({button_x}, {button_y})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Recognition", frame)
    # quit
    if key == ord('q') or close:
        break
    
    

# turn off camera
capture.release()
cv2.destroyAllWindows()
print("camera off")
