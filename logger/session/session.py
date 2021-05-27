from abc import *
from typing import Dict, List, Any, Union

class Session(metaclass=ABCMeta):
    @abstractmethod
    def log_parameters(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metric(self, val_name: str, value: Any) -> None:
        pass

    @abstractmethod
    def log_loss(self, value: float) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass