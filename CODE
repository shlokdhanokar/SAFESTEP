import cv2
import numpy as np
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)  

def beep():
    engine.say("Object Detected!")
    engine.runAndWait()

def detect_objects(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([30, 150, 50])
    upper_bound = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        beep()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame

def main():
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = detect_objects(frame)

        cv2.imshow('Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()
