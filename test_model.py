import torch
import dnnlib
import numpy as np

from training.networks import Generator, ImgEncoder, StyleLipSync_G

img_resolution = 256
img_resolution_log2 = int(np.log2(img_resolution))
block_resolutions = [2**i for i in range(2, img_resolution_log2 + 1)]
print(block_resolutions)
mapping_kwargs = {
    "num_layers": 8,
    "embed_features": None,
    "layer_features": None,
    "activation": "lrelu",
    "lr_multiplier": 0.01,
    "w_avg_beta": 0.995,
}
synthesis_kwargs = {
    "channel_base": 16384 * 2,
    "channel_max": 512,
    "num_fp16_res": 0,
    "conv_clamp": None,
    "architecture": "skip",
    "resample_filter": [1, 3, 3, 1],
    "use_noise": True,
    "activation": "lrelu",
}

g = Generator(
    z_dim=512,  # Input latent (Z) dimensionality.
    c_dim=0,  # Conditioning label (C) dimensionality.
    w_dim=512,  # Intermediate latent (W) dimensionality.
    img_resolution=256,  # Output resolution.
    img_channels=3,  # Number of output color channels.
    mapping_kwargs=mapping_kwargs,  # Arguments for MappingNetwork.
    synthesis_kwargs=synthesis_kwargs,  # Arguments for SynthesisNetwork.
)


for k, v in g.state_dict().items():
    print(k, v.shape)

inp_z = torch.randn(1, 512)
out0 = g(inp_z, inp_z)

encoder = ImgEncoder(
    img_resolution=256,  # Input resolution.
    img_channels=3,  # Number of input color channels.
    architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
    channel_base=32768,  # Overall multiplier for the number of channels.
    channel_max=512,  # Maximum number of channels in any layer.
    num_fp16_res=0,  # Use FP16 for the N highest resolutions.
    conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    block_kwargs={},  # Arguments for DiscriminatorBlock.
)
img = torch.randn(1, 3, 256, 256)
c = torch.randn(1, 512)
out = encoder(img)
for k, v in enumerate(out):
    print(k, v.shape)

g1 = StyleLipSync_G(
    w_dim=512,  # Intermediate latent (W) dimensionality.
    img_resolution=256,  # Output resolution.
    img_channels=3,  # Number of output color channels.
    synthesis_kwargs=synthesis_kwargs,  # Arguments for SynthesisNetwork.
)

out1 = out[::-1] + [None]
out1 = {res: v for res, v in zip(block_resolutions, out1)}
ws = torch.randn(1, 14, 512)
out = g1(ws, out1)
print(out.shape)

