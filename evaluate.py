import argparse
import torch
import matplotlib.pyplot as plt
import os.path as osp

from model import VAE

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser()
# Overall options
parser.add_argument("--checkpoint", type=str, default=osp.sep.join(("checkpoints", "100.pth")), help="Path to the weights")
parser.add_argument("--input_size", default=28*28, help="Input image's size (W x H)")
parser.add_argument("--hidden_size", type=int, default=400, help="Hidden layer's dimenion")
parser.add_argument("--latent_size", type=int, default=20, help="Latent Layer's dimension")

opt = parser.parse_args()
vae = VAE(opt.input_size, opt.hidden_size, opt.latent_size, opt.input_size).to(device)

with torch.no_grad():
    vae.eval()
    temp = torch.load(opt.checkpoint)
    ret = vae.load_state_dict(temp['state_dict'])
    print(ret)

    for i in range(10):
        latent_sample = torch.randn(opt.latent_size).to(device)
        output_size = int(opt.input_size ** 0.5)
        generated_data = vae.decoder(latent_sample).view(output_size, output_size).cpu()

        plt.figure(figsize=(4, 4))
        plt.axis('Off')
        plt.imshow(generated_data, cmap="gray")
        plt.show()
