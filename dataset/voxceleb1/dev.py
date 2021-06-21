import torch
import math
import os
from torch.utils.data import Dataset
from typing import List, Tuple
from ...parameter import DatasetParams
from .extractor import VoxCeleb1Data

class Vox1DevDataset(Dataset):
    files: List[str]
    spk_file_pairs: List[Tuple[int, str]]
    data: VoxCeleb1Data
    params: DatasetParams

    def __init__(self, path: str, data: VoxCeleb1Data, params: DatasetParams) -> None:
        """
        Create a instance and load all files.
        """
        files = []
        # Save (spk_index, file_path) pairs to get features by index at __getitem__
        spk_file_pairs = []

        sub_dirs = os.listdir(path)
        for index, sub in enumerate(sub_dirs):
            for (current, child_dirs, child_files) in os.walk(f"{path}/{sub}"):
                if len(child_dirs) > 0:
                    continue
                tokens = current.split("/")
                for file_path in list(map(lambda x: f"{tokens[-2]}/{tokens[-1]}/{x}".rstrip(), child_files)):
                    spk_file_pairs.append((index, file_path))
                    files.append(file_path)

        self.files = files
        self.spk_file_pairs = spk_file_pairs
        self.data = data
        self.params = params
    
    def __len__(self) -> int:
        return len(self.spk_file_pairs) * self.params.window_count
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get an item: (Input tensor, one-hot encoding index for speaker identification)
        """
        # Get an input (feature with {DATA_WINDOW_SIZE} frames)
        pairs_index = index // self.params.window_count
        window_index = index % self.params.window_count

        file_name = self.spk_file_pairs[pairs_index][1]
        feature = self.data.load(file_name, "dev")

        # Extend tensor to have at least MIN_FRAME_COUNT frames
        extended = feature.repeat(1, math.ceil(self.params.min_frame_count / feature.size()[0]), 1)

        hopping = (extended.size()[1] - self.params.window_size) // (self.params.window_count - 1)
        window_start = hopping * window_index
        x = torch.narrow(extended, 1, window_start, self.params.window_size)

        # Get a label (one-hot encoding index for speaker identification)
        y = self.spk_file_pairs[pairs_index][0]

        return x, y