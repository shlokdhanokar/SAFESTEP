import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech rate (words per minute)

# HSV color thresholds for obstacle detection
LOWER_HSV = np.array([30, 150, 50])
UPPER_HSV = np.array([255, 255, 180])


def alert(message="Object Detected!"):
    """Speak an audio alert to notify the user of an obstacle."""
    engine.say(message)
    engine.runAndWait()


def detect_objects(frame):
    """
    Detect objects in a video frame using HSV color thresholding.

    Args:
        frame: BGR image from the camera feed.

    Returns:
        Processed frame with bounding boxes drawn around detected objects.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        alert()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def main():
    """Main loop: capture video frames, detect objects, and display results."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("SafeStep is running. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        processed_frame = detect_objects(frame)

        cv2.imshow('SafeStep - Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("SafeStep stopped.")


if __name__ == "__main__":
    main()
