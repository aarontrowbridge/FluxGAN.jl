using FluxGAN
using FileIO
using Flux
using CUDA

n = 25

# FIXME: loading errs here
model = load("models/cifar10_goodfellow_MLP_n_$n.jld2")["model"]

layout = (4, 3)

output_dir = "images/CIFAR10"

info = ["loaded", "n", n]

save_image_grid(model, output_dir, layout=layout, file_info=info)