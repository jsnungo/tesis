import torch 
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, in_out_dim, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        # Encoder
        self.fc1 = nn.Linear(in_out_dim, 400)  # Input (flattened image) to hidden
        self.fc2 = nn.Linear(400, 400)  # Input (flattened image) to hidden
        self.fc21 = nn.Linear(400, latent_dim)  # Hidden to mean
        self.fc22 = nn.Linear(400, latent_dim)  # Hidden to log-variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)  # Latent to hidden
        self.fc31 = nn.Linear(400, 400)  # Latent to hidden

        self.fc4 = nn.Linear(400, in_out_dim)  # Hidden to output (reconstruction)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from standard normal distribution
        return eps * std + mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc31(h3))
        return torch.sigmoid(self.fc4(h3))  # Sigmoid for image pixel values

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar