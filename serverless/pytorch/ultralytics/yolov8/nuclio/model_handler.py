import os
from ultralytics import YOLO
import torch
from PIL import Image

class ModelHandler:
    def __init__(self, logger):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Find the model file in the current directory
        model_file = self._find_model_file(".")
        if not model_file:
            self.logger.error("Could not find a .pt model file in the function directory.")
            raise RuntimeError("Model file not found")

        self.logger.info(f"Loading model from {model_file} to {self.device}...")
        self.model = YOLO(model_file)
        self.model.to(self.device)
        self.logger.info("Model loaded successfully.")

    def _find_model_file(self, directory):
        for f in os.listdir(directory):
            if f.endswith(".pt"):
                return os.path.join(directory, f)
        return None

    def handle(self, image, logger):
        logger.info(f"Processing image with size: {image.size}")
        # Use a default confidence threshold, can be overridden by request
        results = self.model(image, conf=0.15)
        logger.info(f"Raw results: {results}")

        detections = []
        for result in results:
            for box in result.boxes:
                xtl, ytl, xbr, ybr = box.xyxy[0].tolist()
                label_id = int(box.cls[0])
                label = self.model.names[label_id]
                confidence = float(box.conf[0])
                detections.append({
                    "label": label,
                    "points": [xtl, ytl, xbr, ybr],
                    "confidence": confidence,
                    "type": "rectangle"
                })
        logger.info(f"Detections: {detections}")
        return detections

    def infer(self, image, threshold):
        self.logger.info(f"Starting inference with threshold {threshold}...")
        results = self.model.predict(image, conf=threshold, verbose=False)

        detections = []
        result = results[0] # The result object from ultralytics

        # Check if the model produced segmentation masks
        has_masks = result.masks is not None

        # Iterate through each detected object
        for i in range(len(result.boxes)):
            box = result.boxes[i]

            # --- Common properties ---
            class_id = int(box.cls)
            label = self.model.names[class_id]
            confidence = float(box.conf)

            # --- Remapping logic ---
            if label in ['bowl','donut']:
                label = 'plate'

            # --- Shape-specific logic ---
            if has_masks:
                # For segmentation, extract the polygon points from the mask
                # The .xy attribute contains the polygon coordinates
                polygon_points = result.masks.xy[i].flatten().tolist()
                detections.append({
                    "label": label,
                    "points": polygon_points,
                    "confidence": confidence,
                    "type": "polygon"
                })
            else:
                # For detection, extract the bounding box
                points = box.xyxy[0].tolist()
                detections.append({
                    "label": label,
                    "points": points,
                    "confidence": confidence,
                    "type": "rectangle"
                })

        self.logger.info(f"Inference complete. Found {len(detections)} objects.")
        return detections
