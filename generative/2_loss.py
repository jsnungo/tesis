import torch
import torch.nn.functional as F


def VAE_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def get_loss_function(name_model: str):
    loss_func = None

    if name_model == 'VAE':
        loss_func = VAE_loss_function

    return loss_func
