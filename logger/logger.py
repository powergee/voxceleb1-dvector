from logger.session.session import Session
from neptune import init
from .session import *
from typing import List, Dict, Any

class Logger(Session):
    sessions : List[Session]

    def __init__(self, *args: List[Session]) -> None:
        self.sessions = args

    def log_parameters(self, params: Dict[str, Any]) -> None:
        for sess in self.sessions:
            if sess.is_available():
                sess.log_parameters(params)

    def log_metric(self, val_name: str, value: Any) -> None:
        for sess in self.sessions:
            if sess.is_available():
                sess.log_metric(val_name, value)

    def log_loss(self, value: float) -> None:
        for sess in self.sessions:
            if sess.is_available():
                sess.log_loss(value)

    def close(self) -> None:
        for sess in self.sessions:
            if sess.is_available():
                sess.close()

    def is_available(self) -> bool:
        result = False
        for sess in self.sessions:
            result |= sess.is_available()
        return result