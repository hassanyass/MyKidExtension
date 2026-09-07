"""
Base Vision Model Abstraction.

Defines the interface for all AI vision models used by MyKid.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np

from packages.shared.types import Detection
from packages.shared.config import MyKidConfig


class VisionModel(ABC):
    """
    Abstract base class for all MyKid vision models.
    
    This interface isolates the rest of the system from specific ML library
    dependencies (like Ultralytics, PyTorch, ONNX, etc).
    """
    
    def __init__(self, config: MyKidConfig):
        """
        Initialize the vision model.
        
        Args:
            config: The global MyKid configuration.
        """
        self.config = config
        
    @abstractmethod
    def load(self) -> None:
        """
        Load the model weights into memory/VRAM.
        
        Raises:
            ModelError: If the model cannot be loaded.
        """
        pass
        
    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Detection]:
        """
        Run inference on an RGB image and return standardized detections.
        
        Implementations MUST filter the raw outputs and ONLY return detections 
        that are considered safety-relevant according to the `shared.labels` mapping,
        and that exceed the configured confidence threshold.
        
        Args:
            image: Preprocessed RGB numpy array (H, W, 3).
            
        Returns:
            A list of Detection objects. Empty list if nothing relevant found.
            
        Raises:
            InferenceError: If inference fails.
        """
        pass
