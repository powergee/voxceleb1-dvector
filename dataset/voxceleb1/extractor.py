import torch
import torch.nn as nn
import torchaudio
import os
import hashlib
from torchaudio.transforms import MelSpectrogram, MFCC
from ...parameter import WaveformParams, AudioPreprocessing
from tqdm import tqdm
from typing import Any, Dict

def get_transforms(params: WaveformParams):
    if params.preprocessing == AudioPreprocessing.MELSPECTOGRAM:
        return MelSpectrogram(**params.kwargs)
    elif params.preprocessing == AudioPreprocessing.MELSPECTOGRAM:
        return MFCC(**params.kwargs)
    else:
        return None

def hash_string(plain: str) -> str:
    h = hashlib.sha1(plain.encode("utf-8")).hexdigest()
    if len(h) > 20:
        return h[:20]
    else:
        return h

class VoxCeleb1Data:
    transforms: nn.Module = None
    vox1_root: str = None
    preload_root: str = None
    param_hash: str = None

    def __init__(self, vox1_root: str, preload_root: str, params: WaveformParams) -> None:
        self.preload_root = preload_root
        self.transforms = get_transforms(params)
        self.vox1_root = vox1_root

        param_hash = params.get_hash()
        self.param_hash = param_hash

        if not os.path.isdir(preload_root):
            os.mkdir(preload_root)

        if not os.path.isdir(os.path.join(preload_root, param_hash)):
            print(".pt files are not found. Extracting...")
            os.mkdir(os.path.join(preload_root, param_hash))

            plist = []
            for root, _, files in os.walk(vox1_root):
                for file in files:
                    if file[-4:] == ".wav":
                        path = os.path.join(root, file)
                        plist.append(path)

            for path in tqdm(plist):
                feature = self.__extract_from_path(path)
                torch.save(feature, os.path.join(preload_root, param_hash, f"{hash_string(path)}.pt"))

    def __extract_from_path(self, path: str) -> torch.Tensor:
        '''
        Load a single wav file and extract features.
        '''
        waveform, _ = torchaudio.load(path)
        features: torch.Tensor = self.transforms(waveform)
        features.unsqueeze(0)

        # size: [1, bins, time]
        return features

    def load(self, file_name: str, kind: str) -> torch.Tensor:
        '''
        Load an extracted tensor from preload directory.
        '''
        if self.preload_root == None:
            raise Exception("preload_root must be set")
        
        path = f"{self.vox1_root}/vox1_{kind}_wav/wav/{file_name}"
        return torch.load(os.path.join(self.preload_root, self.param_hash, f"{hash_string(path)}.pt"))