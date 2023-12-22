import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, z):
        z = F.tanh(self.linear1(z))
        y = F.sigmoid(self.linear2(z))
        return torch.distributions.Normal(y, torch.ones_like(y))


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_var = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        h = F.tanh(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        q_z = torch.distributions.Normal(loc=mu, scale=torch.exp(log_var))
        return q_z

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, output_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_size=hidden_size, output_size=output_size)

    def forward(self, x):
        q_z = self.encoder(x)
        z = q_z.rsample()
        p_x = self.decoder(z)
        return p_x, q_z