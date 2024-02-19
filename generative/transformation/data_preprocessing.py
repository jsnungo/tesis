from .transforms_wav import (LoadAudio, ChangeAmplitude, ChangeSpeedAndPitchAudio,
                            FixAudioLength, ToTensor)


from torchvision.transforms import Compose
import torch
import numpy as np


class OutGenAI(object):
    """Loads an audio into a numpy array."""

    def __init__(self, scale: bool):
        self.scale = scale
        
    def __call__(self, data):
        audio = data['samples']
        x = np.clip(audio, -1, 1)
        if self.scale:
            x = (audio + 1) / 2

        data['samples'] = x

        return data

def get_data_composing(model: str):

    base = [
    LoadAudio(),
    ChangeAmplitude(), 
    ChangeSpeedAndPitchAudio(), 
    FixAudioLength()
    ]

    if model == 'VAE':
        base.append(OutGenAI(True))
    if model == 'DIFFUSSION':
        base.append(OutGenAI(False))


    compose_pp = Compose(base)

    return compose_pp