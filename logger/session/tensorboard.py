from tensorboardX import SummaryWriter
from typing import Dict, List, Any, Union
from os.path import isfile, basename

from logger.session import Session

class TensorboardSession(Session):
    writer: SummaryWriter
    available: bool = True

    def __init__(self, source_paths: List[str]) -> None:
        self.writer = SummaryWriter()
        source_md = []
        for path in source_paths:
            if isfile(path):
                fs = open(path, mode="r")
                source_md.append(f"* {basename(path)}\n\n\"\"\"python")
                source_md.append(fs.read())
                source_md.append("\"\"\"\n\n")
                fs.close()
            else:
                print(f"CometSession: Warning, No such file - {path}")
        self.writer.add_text("Source codes", "\n".join(source_md))
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        self.writer.add_hparams(params)
    
    def log_metric(self, val_name: str, value: Any) -> None:
        self.writer.add_scalar(val_name, value)
    
    def log_loss(self, loss: float) -> None:
        self.log_metric("loss", loss)

    def close(self) -> None:
        self.writer.close()
        self.available = False
    
    def is_available(self) -> bool:
        return self.available