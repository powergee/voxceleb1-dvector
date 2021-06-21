from comet_ml import Experiment
from typing import Dict, List, Any
from os.path import isfile, basename

from logger.session import Session

class CometSession(Session):
    com_ex: Experiment
    available: bool = True

    def __init__(self, source_paths: List[str], **kwargs) -> None:
        try:
            self.com_ex = Experiment(**kwargs)
            for path in source_paths:
                if isfile(path):
                    fs = open(path, mode="r")
                    self.com_ex.log_code(code_name=basename(path), code=fs)
                    fs.close()
                else:
                    print(f"CometSession: Warning, No such file - {path}")
        except Exception as e:
            print("CometSession: Failed to initialize.\n")
            print(e)
            self.available = False
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        try:
            self.com_ex.log_parameters(params)
        except Exception as e:
            print("CometSession: Failed to log parameters.\n")
            print(e)
            self.available = False
    
    def log_metric(self, val_name: str, value: Any) -> None:
        try:
            if val_name == "loss":
                return
            self.com_ex.log_metric(val_name, value)
        except Exception as e:
            print("CometSession: Failed to log metrics.\n")
            print(e)
            self.available = False

    def log_loss(self, loss: float) -> None:
        if self.com_ex.auto_metric_logging:
            # If comet is already set to log loss automatically, just return.
            return
        else:
            self.log_metric("loss", loss)

    def close(self) -> None:
        self.com_ex.end()

    def is_available(self) -> bool:
        return self.available