from ultralytics import YOLOWorld


class OpenVocabularyDetector:

    def __init__(self):

        print("Loading Open-Vocabulary AI...")

        self.model = YOLOWorld("yolov8s-world.pt")

        self.classes = [
            # People
            "person",
            "man",
            "woman",
            "child",

            # Vehicles
            "car",
            "bus",
            "truck",
            "van",
            "motorcycle",
            "bicycle",
            "scooter",
            "airplane",
            "helicopter",
            "boat",
            "ship",
            "train",

            # Animals
            "dog",
            "cat",
            "horse",
            "cow",
            "bird",
            "elephant",
            "lion",
            "tiger",

            # Musical instruments
            "guitar",
            "piano",
            "keyboard",
            "drum",
            "violin",
            "microphone",
            "headset",
            "headphones",
            "earphones",

            # Gym and fitness equipment
            "dumbbell",
            "barbell",
            "weight plate",
            "kettlebell",
            "bench press",
            "gym bench",
            "squat rack",
            "power rack",
            "treadmill",
            "exercise bike",
            "rowing machine",
            "cable machine",
            "pull-up bar",
            "gym equipment",
            "hospital bed",
            "medical equipment",
            "police uniform",
            "school desk",
            "classroom",
            "restaurant table",
            "kitchen counter",
            "park bench",
            "beach",
            "forest",

            # Electronics
            "cell phone",
            "laptop",
            "computer",
            "tablet",
            "camera",
            "television",

            # Furniture
            "chair",
            "table",
            "sofa",
            "bed",
            "desk",

            # Bags / personal objects
            "backpack",
            "handbag",
            "suitcase",
            "bag",
            "wallet",

            # Clothing
            "shirt",
            "t-shirt",
            "dress",
            "jacket",
            "pants",
            "skirt",
            "hat",
            "shoe",

            # Food
            "apple",
            "banana",
            "orange",
            "watermelon",
            "pizza",
            "burger",
            "cake",

            # Vegetables
            "carrot",
            "tomato",
            "potato",
            "onion",

            # Common objects
            "bottle",
            "cup",
            "glass",
            "plate",
            "bowl",
            "fork",
            "spoon",
            "knife",
            "pot",
            "pan",
            "frying pan",
            "cooking utensil",
            "food",
            "book",
            "umbrella",
            "clock",
            "ball",
            "door",
            "window"
        ]

        self.model.set_classes(self.classes)

        print("Open-Vocabulary AI ready!")


    def detect(self, frame, confidence=0.25):

        results = self.model.predict(
            frame,
            conf=confidence,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence_score = float(
                    box.conf[0]
                )

                class_name = self.model.names[
                    class_id
                ]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                detections.append({

                    "label": class_name,

                    "confidence": confidence_score,

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                })

        return detections