"""Neighbor-based Longitudinal Completion (NLC) for photonic Wannier spaces."""

from .completion import prepare_longitudinal_bundle
from .models import NLCCompletionArtifacts, NLCCompletionResult, NLCSettings

__all__ = [
    "NLCSettings",
    "NLCCompletionResult",
    "NLCCompletionArtifacts",
    "prepare_longitudinal_bundle",
]
