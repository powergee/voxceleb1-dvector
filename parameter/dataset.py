from .abc import ABCParams

class DatasetParams(ABCParams):
    min_frame_count: int
    window_count: int
    window_size: int