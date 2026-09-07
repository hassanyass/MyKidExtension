"""
MyKid Image Pipeline.

Orchestrates the end-to-end processing of images through all MyKid engines.
"""
import logging
from typing import Union, Tuple
import numpy as np
import time

from packages.shared.config import MyKidConfig
from packages.shared.types import AnalysisResult, ProcessingState
from packages.shared.logger import get_logger, log_event, LogEvent
from packages.shared.errors import MyKidError

from packages.image_processing.loader import load_image
from packages.image_processing.preprocessor import resize_preserve_aspect_ratio
from packages.vision.yolo_model import YoloModel
from packages.risk.risk_engine import RiskEngine
from packages.protection.protection_engine import ProtectionEngine


logger = get_logger("mykid.pipeline.image")


class ImagePipeline:
    """
    End-to-end image processing pipeline.
    Combines loading, preprocessing, detection, risk assessment, and protection.
    """
    
    def __init__(self, config: MyKidConfig, vision_model=None):
        """
        Initialize the pipeline with all necessary engines.
        
        Args:
            config: Global configuration object.
            vision_model: Optional pre-instantiated vision model (useful for testing).
                          If None, YoloModel will be instantiated.
        """
        self.config = config
        
        # Initialize sub-components
        self.vision = vision_model if vision_model else YoloModel(config.model.object_detection)
        self.risk = RiskEngine(config)
        self.protection = ProtectionEngine(config)
        
        # Ensure vision model is loaded
        try:
            self.vision.load()
        except Exception as e:
            log_event(logger, logging.ERROR, LogEvent.ERROR_INFERENCE, f"Failed to load vision model: {e}")
            raise

    def process(self, source: Union[str, bytes, np.ndarray]) -> Tuple[np.ndarray, AnalysisResult]:
        """
        Process an image from start to finish.
        
        Args:
            source: Image path, raw bytes, or numpy array.
            
        Returns:
            Tuple of (Protected Image Array, AnalysisResult).
        """
        start_time = time.time()
        log_event(
            logger, 
            logging.INFO, 
            LogEvent.IMAGE_ANALYSIS_STARTED, 
            "Started processing image"
        )
        
        try:
            # 1. Load
            image = load_image(source)
            log_event(logger, logging.DEBUG, LogEvent.IMAGE_LOADED, "Image loaded successfully")
            
            # 2. Preprocess
            processed_image = resize_preserve_aspect_ratio(image, self.config.image.max_size)
            img_shape = processed_image.shape[:2] # (height, width)
            log_event(logger, logging.DEBUG, LogEvent.IMAGE_PROCESSED, f"Image preprocessed to {img_shape}")
            
            # 3. Vision Detection
            detections = self.vision.predict(processed_image)
            log_event(
                logger, 
                logging.DEBUG, 
                LogEvent.DETECTION_COMPLETED, 
                f"Vision engine found {len(detections)} objects"
            )
            
            # 4. Risk Assessment
            analysis = self.risk.evaluate(detections=detections, image_shape=img_shape)
            
            # 5. Protection
            final_image = self.protection.protect(processed_image, analysis)
            
            elapsed = time.time() - start_time
            log_event(
                logger, 
                logging.INFO, 
                LogEvent.IMAGE_ANALYSIS_COMPLETED, 
                f"Image processing completed in {elapsed:.3f}s",
                metadata={"elapsed_seconds": elapsed, "final_risk": analysis.overall_risk.value}
            )
            
            return final_image, analysis
            
        except MyKidError as e:
            # Re-raise known pipeline errors
            elapsed = time.time() - start_time
            log_event(
                logger, 
                logging.ERROR, 
                LogEvent.PROCESSING_ERROR, 
                f"Pipeline error: {e.message}",
                metadata={"elapsed_seconds": elapsed, "error_code": e.error_code}
            )
            raise
        except Exception as e:
            # Wrap unknown errors
            elapsed = time.time() - start_time
            log_event(
                logger, 
                logging.ERROR, 
                LogEvent.PROCESSING_ERROR, 
                f"Unexpected pipeline error: {str(e)}",
                metadata={"elapsed_seconds": elapsed}
            )
            raise
