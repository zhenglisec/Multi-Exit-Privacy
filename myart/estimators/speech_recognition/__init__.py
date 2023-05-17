"""
Module containing estimators for speech recognition.
"""
from myart.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin

from myart.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from myart.estimators.speech_recognition.pytorch_espresso import PyTorchEspresso
from myart.estimators.speech_recognition.tensorflow_lingvo import TensorFlowLingvoASR
