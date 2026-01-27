# how we call each module to run a specific task
from .diagnose import Diagnose
from .detect import DetectLabel
from .detect import DetectFeature
__all__ = [
    'Diagnose', 'DetectLabel', 'DetectFeature'
]