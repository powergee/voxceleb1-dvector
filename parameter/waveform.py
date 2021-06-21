from typing import Any, Dict
from .abc import ABCParams
from enum import Enum, auto

class AudioPreprocessing(Enum):
    MELSPECTOGRAM = auto()
    MFCC = auto()

class WaveformParams(ABCParams):
    preprocessing: AudioPreprocessing
    kwargs: Dict[str, Any]