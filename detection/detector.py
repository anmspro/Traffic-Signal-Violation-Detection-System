"""
Lazy-Loading YOLOv3 Detector for Traffic Violation Detection

This module provides a lazy-loading wrapper for the YOLOv3 model with GPU
configuration and detection capabilities.
"""

import time
import logging
import config
from models import make_yolov3_model, WeightReader


# ============================================================================
# GPU Configuration (must be done before model loading)
# ============================================================================

# Configure GPU before any TensorFlow operations
_gpus = config.configure_gpu()
if _gpus:
    logging.info(f"YOLOv3 will use GPU acceleration: {_gpus}")
else:
    logging.info("YOLOv3 will run on CPU")


# ============================================================================
# YOLOv3 Violation Detector with Lazy Loading
# ============================================================================

class ViolationDetector:
    """
    Lazy-loading wrapper for YOLOv3 traffic violation detection model.

    The model is not loaded until the first prediction is made,
    improving startup time for non-processing operations.

    Attributes:
        _model: Cached Keras model (None until first use)
        _load_time: Time taken to load model (seconds)

    Performance:
        - First prediction: ~30s (model loading + inference)
        - Subsequent predictions: ~0.1s (inference only)

    Example:
        >>> detector = ViolationDetector()
        >>> # Model not loaded yet
        >>> image = preprocess_input(frame, config.NET_H, config.NET_W)
        >>> predictions = detector.predict(image)  # Model loads here
        >>> predictions2 = detector.predict(image2)  # Uses cached model
    """

    def __init__(self):
        """
        Initialize detector without loading model.

        The model will be loaded lazily on first predict() call.
        """
        self._model = None
        self._load_time = None
        logging.info("ViolationDetector initialized (lazy loading enabled)")

    @property
    def model(self):
        """
        Get the YOLOv3 model, loading it on first access.

        This property implements lazy loading - the expensive model
        loading operation only happens on first access.

        Returns:
            keras.Model: Loaded YOLOv3 model ready for inference

        Raises:
            FileNotFoundError: If weights file doesn't exist
            RuntimeError: If model loading fails
        """
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """
        Internal method to load YOLOv3 model and weights.

        Raises:
            FileNotFoundError: If WEIGHTS_PATH doesn't exist
            RuntimeError: If model creation or weight loading fails
        """
        logging.info("Loading YOLOv3 model...")
        start_time = time.time()

        try:
            # Verify weights file exists
            if not config.WEIGHTS_PATH.exists():
                raise FileNotFoundError(
                    f"YOLOv3 weights not found at {config.WEIGHTS_PATH}. "
                    "Please download from: https://pjreddie.com/media/files/yolov3.weights"
                )

            # Create model architecture
            logging.info("Building YOLOv3 architecture...")
            self._model = make_yolov3_model()

            # Load pre-trained weights
            logging.info(f"Loading weights from {config.WEIGHTS_PATH}...")
            weight_reader = WeightReader(str(config.WEIGHTS_PATH))
            weight_reader.load_weights(self._model)

            # Calculate load time
            self._load_time = time.time() - start_time

            logging.info(f"✅ Model loaded successfully in {self._load_time:.2f}s")
            logging.info(f"Model parameters: {self._model.count_params():,}")

        except Exception as e:
            logging.error(f"Failed to load YOLOv3 model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def predict(self, image):
        """
        Run object detection on preprocessed image.

        Args:
            image (np.ndarray): Preprocessed image with shape (1, NET_H, NET_W, 3)

        Returns:
            list: YOLOv3 predictions for three scales
                  [output_1, output_2, output_3]

        Raises:
            ValueError: If image has wrong shape
            RuntimeError: If prediction fails

        Example:
            >>> preprocessed = preprocess_input(frame, config.NET_H, config.NET_W)
            >>> predictions = detector.predict(preprocessed)
            >>> boxes = decode_netout(predictions[0], ...)
        """
        # Validate input shape
        expected_shape = (1, config.NET_H, config.NET_W, 3)
        if image.shape != expected_shape:
            raise ValueError(
                f"Expected image shape {expected_shape}, got {image.shape}"
            )

        try:
            # This will trigger lazy loading if needed
            return self.model.predict(image, verbose=0)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Detection failed: {e}") from e

    @property
    def is_loaded(self):
        """Check if model is currently loaded in memory."""
        return self._model is not None

    def unload(self):
        """
        Unload model from memory to free resources.

        Useful for long-running applications that process videos
        in batches with idle time between batches.
        """
        if self._model is not None:
            logging.info("Unloading model from memory")
            del self._model
            self._model = None

            # Force garbage collection
            import gc
            gc.collect()


# ============================================================================
# Global Detector Instance (Lazy Loading)
# ============================================================================

# Create global detector instance (model will load on first predict() call)
detector = ViolationDetector()
