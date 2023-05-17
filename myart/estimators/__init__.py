"""
This module contains the Estimator API.
"""
from myart.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
    DecisionTreeMixin,
)

from myart.estimators.keras import KerasEstimator
from myart.estimators.mxnet import MXEstimator
from myart.estimators.pytorch import PyTorchEstimator
from myart.estimators.scikitlearn import ScikitlearnEstimator
from myart.estimators.tensorflow import TensorFlowEstimator, TensorFlowV2Estimator

from myart.estimators import certification
from myart.estimators import classification
from myart.estimators import encoding
from myart.estimators import generation
from myart.estimators import object_detection
from myart.estimators import poison_mitigation
from myart.estimators import regression
from myart.estimators import speech_recognition
