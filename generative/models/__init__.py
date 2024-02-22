from .vae import VAE
from .WaveNet import WaveNet_Speech_Commands as WaveNet

available_models = [
    'VAE', 'DIFFUSION'
]

def create_model(model_name, general_config=None):
    if model_name == "VAE":
        model = VAE()
    elif model_name == "DIFFUSION":
        net_config = general_config['wavenet_config']
        model = WaveNet(**net_config)

    return model