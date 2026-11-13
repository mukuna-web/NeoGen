"""
NeoGen Advisory Board — Virtual Biotech Organization

A multi-department AI agent system that provides expert consultation
and multi-layer review for NeoGen drug prediction platform development.

Usage:
    python -m neogen.advisory "How do I improve clearance prediction?"
    python -m neogen.advisory --review "Add pKa features to bioavailability model"
"""

from .orchestrator import AdvisoryBoard

__all__ = ["AdvisoryBoard"]
