"""
Weight Reader for YOLOv3 Darknet Format

Utilities for loading pre-trained YOLOv3 weights from Darknet format into Keras models.
"""

import struct
import logging
import numpy as np


class WeightReader:
    """
    Reader for loading YOLOv3 pre-trained weights from Darknet format.

    This class reads binary weight files in the Darknet format (.weights)
    and loads them into a Keras model. Handles version compatibility and
    proper weight ordering for convolutional and batch normalization layers.

    Attributes:
        offset (int): Current position in the weight array
        all_weights (np.ndarray): All weights loaded from file as float32

    Example:
        >>> model = make_yolov3_model()
        >>> weight_reader = WeightReader('yolov3.weights')
        >>> weight_reader.load_weights(model)
        >>> # Model is now ready for inference
    """
    def __init__(self, weight_file):
        """
        Initialize weight reader and load weights from file.

        Args:
            weight_file (str): Path to Darknet weights file (.weights)

        Raises:
            FileNotFoundError: If weight file doesn't exist
            struct.error: If file format is invalid
        """
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)

            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        """
        Read specified number of weights from current position.

        Args:
            size (int): Number of weight values to read

        Returns:
            np.ndarray: Array of weight values
        """
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model):
        """
        Load pre-trained weights into YOLOv3 Keras model.

        Iterates through all 106 convolutional layers in YOLOv3 and loads
        weights from the Darknet format, handling both batch normalization
        and bias parameters correctly.

        Args:
            model (keras.Model): YOLOv3 model created by make_yolov3_model()

        Note:
            Layers 81, 93, and 105 are detection layers without batch norm.
            Weights are transposed from Darknet [out, in, h, w] to
            Keras [h, w, in, out] format.

        Example:
            >>> model = make_yolov3_model()
            >>> reader = WeightReader('yolov3.weights')
            >>> reader.load_weights(model)
        """
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                logging.debug(f"Loading weights of convolution #{i}")

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta  = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance

                    weights = norm_layer.set_weights([gamma, beta, mean, var])

                if len(conv_layer.get_weights()) > 1:
                    bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                logging.debug(f"No convolution #{i}")

    def reset(self):
        """Reset read offset to beginning of weight array."""
        self.offset = 0
