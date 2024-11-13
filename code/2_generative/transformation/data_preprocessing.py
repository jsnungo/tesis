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
        if self.scale:
            x = np.clip(audio, -1, 1)
        x = np.expand_dims(x, axis=0)
        data['samples'] = x

        return data

def get_data_composing(model: str, config: dict):
    '''
    Get the data composing for the model

    Args:
    model (str): The model name
    config (dict): The configuration dictionary

    Returns:
    compose_pp (Compose): The data composing
    '''

    sr = config['audio_config']['sample_rate']
    base = [
        LoadAudio(sample_rate=sr),
        # ChangeAmplitude(), 
        # ChangeSpeedAndPitchAudio(), 
        FixAudioLength()
    ]

    if model == 'VAE':
        base.append(OutGenAI(True))
    if model == 'DIFFUSION':
        base.append(OutGenAI(True))


    compose_pp = Compose(base)

    return compose_pp