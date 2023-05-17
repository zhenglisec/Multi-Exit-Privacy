"""
Module providing evasion attacks under a common interface.
"""
from myart.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from myart.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from myart.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
from myart.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from myart.attacks.evasion.adversarial_asr import CarliniWagnerASR
from myart.attacks.evasion.auto_attack import AutoAttack
from myart.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from myart.attacks.evasion.brendel_bethge import BrendelBethgeAttack
from myart.attacks.evasion.boundary import BoundaryAttack
from myart.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod, CarliniL0Method
from myart.attacks.evasion.decision_tree_attack import DecisionTreeAttack
from myart.attacks.evasion.deepfool import DeepFool
from myart.attacks.evasion.dpatch import DPatch
from myart.attacks.evasion.dpatch_robust import RobustDPatch
from myart.attacks.evasion.elastic_net import ElasticNet
from myart.attacks.evasion.fast_gradient import FastGradientMethod
from myart.attacks.evasion.frame_saliency import FrameSaliencyAttack
from myart.attacks.evasion.feature_adversaries.feature_adversaries_numpy import FeatureAdversariesNumpy
from myart.attacks.evasion.feature_adversaries.feature_adversaries_pytorch import FeatureAdversariesPyTorch
from myart.attacks.evasion.feature_adversaries.feature_adversaries_tensorflow import FeatureAdversariesTensorFlowV2
from myart.attacks.evasion.geometric_decision_based_attack import GeoDA
from myart.attacks.evasion.hclu import HighConfidenceLowUncertainty
from myart.attacks.evasion.hop_skip_jump import HopSkipJump
from myart.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
from myart.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
from myart.attacks.evasion.iterative_method import BasicIterativeMethod
from myart.attacks.evasion.lowprofool import LowProFool
from myart.attacks.evasion.newtonfool import NewtonFool
from myart.attacks.evasion.pe_malware_attack import MalwareGDTensorFlow
from myart.attacks.evasion.pixel_threshold import PixelAttack
from myart.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from myart.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from myart.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)
from myart.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)
from myart.attacks.evasion.over_the_air_flickering.over_the_air_flickering_pytorch import OverTheAirFlickeringPyTorch
from myart.attacks.evasion.saliency_map import SaliencyMapMethod
from myart.attacks.evasion.shadow_attack import ShadowAttack
from myart.attacks.evasion.shapeshifter import ShapeShifter
from myart.attacks.evasion.simba import SimBA
from myart.attacks.evasion.spatial_transformation import SpatialTransformation
from myart.attacks.evasion.square_attack import SquareAttack
from myart.attacks.evasion.pixel_threshold import ThresholdAttack
from myart.attacks.evasion.universal_perturbation import UniversalPerturbation
from myart.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
from myart.attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
from myart.attacks.evasion.wasserstein import Wasserstein
from myart.attacks.evasion.zoo import ZooAttack
