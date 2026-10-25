"""FreeRyder: find, benchmark and ride the best free OpenRouter model."""

from .bench import Benchmark, ModelReport, Weights
from .tasks import TASKS, Task, story_tells
from .client import CallResult, OpenRouterClient
from .picker import Config, FreeRyder, NoModelAvailable

__all__ = ["FreeRyder", "Config", "NoModelAvailable", "Benchmark", "ModelReport", "Task",
           "TASKS", "Weights", "story_tells", "OpenRouterClient", "CallResult"]
__version__ = "0.1.0"
