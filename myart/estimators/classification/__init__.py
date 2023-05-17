"""
Classifier API for applying all attacks. Use the :class:`.Classifier` wrapper to be able to apply an attack to a
preexisting model.
"""
from myart.estimators.classification.classifier import (
    ClassifierMixin,
    ClassGradientsMixin,
)

from myart.estimators.classification.blackbox import BlackBoxClassifier, BlackBoxClassifierNeuralNetwork
from myart.estimators.classification.catboost import CatBoostARTClassifier
from myart.estimators.classification.detector_classifier import DetectorClassifier
from myart.estimators.classification.ensemble import EnsembleClassifier
from myart.estimators.classification.GPy import GPyGaussianProcessClassifier
from myart.estimators.classification.keras import KerasClassifier
from myart.estimators.classification.lightgbm import LightGBMClassifier
from myart.estimators.classification.mxnet import MXClassifier
from myart.estimators.classification.pytorch import PyTorchClassifier
from myart.estimators.classification.query_efficient_bb import QueryEfficientGradientEstimationClassifier
from myart.estimators.classification.scikitlearn import SklearnClassifier
from myart.estimators.classification.tensorflow import (
    TFClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
)
from myart.estimators.classification.xgboost import XGBoostClassifier
