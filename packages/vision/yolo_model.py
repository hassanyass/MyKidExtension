"""
YOLO implementation of the VisionModel abstraction.
"""
from pathlib import Path
from typing import List
import numpy as np

import logging
from packages.shared.types import Detection, BoundingBox
from packages.shared.config import MyKidConfig
from packages.shared.errors import ModelError, InferenceError
from packages.shared.labels import is_safety_relevant, get_base_risk
from packages.shared.logger import get_logger, log_event, LogEvent
from packages.vision.base_model import VisionModel


logger = get_logger("mykid.vision.yolo")


class YoloModel(VisionModel):
    """
    Ultralytics YOLO wrapper implementation.
    """
    
    def __init__(self, config: MyKidConfig):
        super().__init__(config)
        self.model = None
        self._conf_threshold = self.config.detection.object_confidence_threshold
        
    def load(self) -> None:
        """
        Load the YOLO model weights.
        """
        try:
            from ultralytics import YOLO
            
            # Determine path
            model_name = self.config.model.object_detection.weights
            model_dir = Path(self.config.model.path)
            
            # If the weights file exists locally in our models dir, use that
            # Otherwise use just the name (ultralytics will download it to current dir)
            local_path = model_dir / model_name
            if local_path.exists():
                load_path = str(local_path)
            else:
                load_path = model_name
                
            self.model = YOLO(load_path)
            log_event(
                logger, 
                logging.INFO,
                LogEvent.MODEL_LOADED, 
                "Model loaded successfully",
                metadata={"model_name": model_name, "path": load_path}
            )
            
        except ImportError as e:
            raise ModelError("Ultralytics library not installed. Run 'pip install ultralytics'") from e
        except Exception as e:
            error_msg = f"Failed to load YOLO model: {str(e)}"
            log_event(logger, logging.ERROR, LogEvent.ERROR_MODEL, error_msg, metadata={"error": str(e)})
            raise ModelError(error_msg) from e
            
    def predict(self, image: np.ndarray) -> List[Detection]:
        """
        Run inference and convert to standardized MyKid Detections.
        """
        if self.model is None:
            raise ModelError("Model not loaded. Call load() first.")
            
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise InferenceError("Invalid or empty image array")
            
        try:
            # Run inference
            # We set verbose=False to keep standard output clean
            results = self.model(image, conf=self._conf_threshold, verbose=False)
            
            # We only pass one image, so we only get one result object
            result = results[0]
            detections = []
            
            if result.boxes is None or len(result.boxes) == 0:
                log_event(logger, logging.INFO, LogEvent.DETECTION_COMPLETED, "Inference completed", metadata={"count": 0, "safety_relevant": 0})
                return detections
                
            # Parse bounding boxes
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = self.model.names[cls_id]
                
                # IMPORTANT: Only process safety-relevant labels
                if not is_safety_relevant(label):
                    continue
                    
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                width = x2 - x1
                height = y2 - y1
                
                bbox = BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(width),
                    height=float(height)
                )
                
                base_risk = get_base_risk(label)
                
                detection = Detection(
                    label=label,
                    confidence=conf,
                    bbox=bbox,
                    risk=base_risk
                )
                
                detections.append(detection)
                
            log_event(
                logger, 
                logging.INFO,
                LogEvent.DETECTION_COMPLETED, 
                "Inference completed",
                metadata={
                    "total_boxes": len(result.boxes), 
                    "safety_relevant": len(detections)
                }
            )
            
            return detections
            
        except Exception as e:
            error_msg = f"YOLO inference failed: {str(e)}"
            log_event(logger, logging.ERROR, LogEvent.ERROR_INFERENCE, error_msg, metadata={"error": str(e)})
            raise InferenceError(error_msg) from e
