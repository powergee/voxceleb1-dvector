from abc import *

class ABCParams(metaclass=ABCMeta):
    @abstractmethod
    def get_hash(self) -> None:
        pass