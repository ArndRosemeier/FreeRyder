"""FreeRyder: find, benchmark and ride the best free OpenRouter model."""

from .bench import TASKS, Benchmark, ModelReport, Task, Weights
from .client import CallResult, OpenRouterClient
from .picker import Config, FreeRyder, NoModelAvailable

__all__ = ["FreeRyder", "Config", "NoModelAvailable", "Benchmark", "ModelReport", "Task",
           "TASKS", "Weights", "OpenRouterClient", "CallResult"]
__version__ = "0.1.0"
