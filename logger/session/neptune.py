from logging import exception
from neptune import new as neptune
from neptune.new.run import Run
from typing import Dict, List, Any, Union

from logger.session import Session

class NeptuneSession(Session):
    nep_ex: Run
    available: bool = True

    def __init__(self, source_paths: List[str], **kwargs) -> None:
        try:
            self.nep_ex = neptune.init(source_files=source_paths, **kwargs)
        except Exception as e:
            print("NeptuneSession: Failed to initialize.\n")
            print(e)
            self.available = False
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        try:
            self.nep_ex["parameters"] = params
        except Exception as e:
            print("NeptuneSession: Failed to log parameters.\n")
            print(e)
            self.available = False
    
    def log_metric(self, val_name: str, value: Any) -> None:
        try:
            self.nep_ex[val_name].log(value)
        except Exception as e:
            print("NeptuneSession: Failed to log metrics.\n")
            print(e)
            self.available = False
    
    def log_loss(self, loss: float) -> None:
        self.log_metric("loss", loss)
    
    def close(self) -> None:
        self.nep_ex.stop()
        self.available = False
    
    def is_available(self) -> bool:
        return self.available