"""Semantic VAD: turn-end detection from speech content.

Mirrors unmute.sh's design: a pause/turn-end probability is smoothed with an
exponential moving average and compared against a threshold. Here the
probability comes from Smart Turn v3 (audio-native), while Silero VAD provides
raw speech-activity segmentation.
"""

from .semantic_vad import SemanticVAD, TurnEndResult

__all__ = ["SemanticVAD", "TurnEndResult"]
