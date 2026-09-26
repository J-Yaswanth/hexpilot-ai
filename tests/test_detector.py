import cv2

from app.vision.detector import VideoObjectDetector


video_path = "data/videos/test.mp4"

detector = VideoObjectDetector()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()


frame_number = 0

print("\n🎯 AI OBJECT TRACKING")
print("=" * 50)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Process every 5th frame
    if frame_number % 5 != 0:
        continue

    detections = detector.track(frame)

    print(f"\nFrame {frame_number}")

    for detection in detections:

        print(
            f"ID: {detection['track_id']} | "
            f"Object: {detection['label']} | "
            f"Confidence: {detection['confidence']:.2f}"
        )


cap.release()

print("\n✅ Tracking test completed.")