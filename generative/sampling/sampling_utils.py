import torch


def get_vae_sample(model, num_samples, device):
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)

        samples = (samples * 2) - 1 # TODO: Revisar que sea necesaria esta inversion

    return samples