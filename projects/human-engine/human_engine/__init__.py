"""human_engine — a psychologically-grounded simulator of a human mind.

Spec: notes/research/design/spec.md
Theory: notes/research/<topic>/theories.md
"""

from .persona import Persona, default_persona
from .state import State
from .engine import Engine
from .llm import MockLLM

__all__ = ["Persona", "default_persona", "State", "Engine", "MockLLM"]
__version__ = "0.1.0"
