from .vae import compute_vae_training_loss

available_models = [
    'VAE', 'DIFFUSSION'
]

def get_loss_train(model_name):
    if model_name == "VAE":
        loss_training_function = compute_vae_training_loss

    return loss_training_function