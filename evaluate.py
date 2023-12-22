import torch
import matplotlib.pyplot as plt

from model import VAE

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

checkpoint = "checkpoints/100.pth"
input_size = 28 * 28
hidden_size = 512
latent_dim = 20
vae = VAE(input_size, hidden_size, latent_dim, input_size).to(device)

with torch.no_grad():
    vae.eval()
    temp = torch.load(checkpoint)
    ret = vae.load_state_dict(temp['state_dict'])
    print(ret)

    latent_sample = torch.randn(latent_dim).to(device)
    generated_data = vae.decoder(latent_sample).view(28, 28).cpu()

    plt.figure(figsize=(4, 4))
    plt.axis('Off')
    plt.imshow(generated_data, cmap="gray")
    plt.show()
