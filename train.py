import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from model import VAE

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

'''
Parameters
'''
num_epochs = 100
batch_size = 256
learning_rate = 1e-3
input_size = 28 * 28
hidden_size = 512
latent_dim = 8

transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize(0.5, 1)
    ])
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_data = DataLoader(mnist, batch_size=batch_size, shuffle=True, pin_memory=True)
vae = VAE(input_size, hidden_size, latent_dim, input_size).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs + 1):
    # Resume
    if os.path.exists("checkpoints/" + str(epoch) + ".pth"):
        temp = torch.load("checkpoints/" + str(epoch) + ".pth")
        ret = vae.load_state_dict(temp['state_dict'])
        print(ret)
        optimizer.load_state_dict(temp['optimizer'])
        continue

    for data in train_data:
        inputs, _ = data
        inputs = inputs.view(-1, input_size).to(device)
        optimizer.zero_grad()
        p_x, q_z = vae(inputs)

        # Loss
        likelihood = p_x.log_prob(inputs).sum(-1).mean()
        kl = torch.distributions.kl_divergence(q_z, torch.distributions.Normal(0, 1.0)).sum(-1).mean()
        loss = -(likelihood - kl)
        loss.backward()
        optimizer.step()
    
    print("Epoch {:<4} Loss: {:<8.4f} Likelihood: {:<8.4f} KL: {:<8.4f}".format(epoch, loss.item(), likelihood.item(), kl.item()))
    checkpoints = {
        'state_dict': vae.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoints, "checkpoints/" + str(epoch) + ".pth")
