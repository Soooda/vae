import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp

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
batch_size = 128
learning_rate = 1e-3
input_size = 28 * 28
hidden_size = 400
latent_dim = 20

transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize(0.5, 1)
    ])
mnist = torchvision.datasets.MNIST(root=osp.abspath('./data'), train=True, download=True, transform=transform)
train_data = DataLoader(mnist, batch_size=batch_size, shuffle=True, pin_memory=True)
vae = VAE(input_size, hidden_size, latent_dim, input_size).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

def loss_fn(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    KL = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    return BCE + KL


for epoch in range(1, num_epochs + 1):
    # Resume
    if osp.exists(osp.join("checkpoints/" + str(epoch) + ".pth")):
        temp = torch.load(osp.join("checkpoints/" + str(epoch) + ".pth"))
        ret = vae.load_state_dict(temp['state_dict'])
        print(ret)
        optimizer.load_state_dict(temp['optimizer'])
        continue

    for data in train_data:
        inputs, _ = data
        inputs = inputs.view(-1, input_size).to(device)
        optimizer.zero_grad()
        recon_inputs, mu, log_var = vae(inputs)

        # Loss
        loss = loss_fn(recon_inputs, inputs, mu, log_var)
        loss.backward()
        optimizer.step()
    
    print("Epoch {:<4} Loss: {:<8.4f}".format(epoch, loss.item()))
    checkpoints = {
        'state_dict': vae.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoints, osp.join("checkpoints/" + str(epoch) + ".pth"))
