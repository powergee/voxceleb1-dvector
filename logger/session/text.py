from typing import Dict, List, Any
from datetime import datetime
from os import path
from io import TextIOWrapper
import shutil

from logger.session import Session

def trim_common_dirs(paths: List[str]) -> List[str]:
    splited: List[List[str]] = []
    for i in range(len(paths)):
        real_path = path.realpath(paths[i])
        splited.append(real_path.split(" "))
    
    common = 0
    max_len = max(map(len, splited))

    for j in range(max_len):
        all_same = True
        for i in range(len(splited)-1):
            if splited[i][j] != splited[i+1][j]:
                all_same = False
                break
        if all_same:
            common += 1
        else:
            break

    result: List[str] = []
    for i in range(len(splited)):
        result.append("/".join(splited[i][:common]))
    return result


class TextSession(Session):
    archive_path: str
    available: bool = True
    writer: TextIOWrapper
    loss_count: int = 0
    loss_sum: int = 0

    def print_on_writer(self, line: str) -> None:
        self.writer.write(f"[{datetime.now().strftime('%c')}] ${line}\n")
        self.writer.flush()

    def __init__(self, source_paths: List[str], archive_root: str) -> None:
        try:
            source_paths = trim_common_dirs(source_paths)
            time_str = datetime.now().strftime('%c')
            self.archive_path = path.join(archive_root, time_str)

            for p in source_paths:
                if path.isfile(p):
                    shutil.copy(p, path.join(self.archive_path, path.basename(p)))
                else:
                    print("TextSession: Warning, No such file - {p}")
            
            self.writer = open(path.join(self.archive_path, "log.txt"))
            self.print_on_writer("Initialized.")
        except Exception as e:
            print("TextSession: Failed to initialize.\n")
            print(e)
            self.available = False
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        try:
            self.print_on_writer(f"Parameters: {params}")
        except Exception as e:
            print("TextSession: Failed to log parameters.\n")
            print(e)
            self.available = False
    
    def log_metric(self, val_name: str, value: Any) -> None:
        try:
            if val_name == "loss":
                self.loss_sum += value
                self.loss_count += 1
                if self.loss_count == 1000:
                    self.print_on_writer(f"avg loss in 1000 iterations: {self.loss_sum / self.loss_count}")
                    self.loss_count = 0
                    self.loss_sum = 0
            else:
                self.print_on_writer(f"{val_name}: {value}")
        except Exception as e:
            print("TextSession: Failed to log metrics.\n")
            print(e)
            self.available = False

    def close(self) -> None:
        self.available = False
        self.writer.close()

    def is_available(self) -> bool:
        return self.available