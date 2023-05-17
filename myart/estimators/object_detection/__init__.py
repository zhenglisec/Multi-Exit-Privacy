"""
Module containing estimators for object detection.
"""
from myart.estimators.object_detection.object_detector import ObjectDetectorMixin

from myart.estimators.object_detection.python_object_detector import PyTorchObjectDetector
from myart.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from myart.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
