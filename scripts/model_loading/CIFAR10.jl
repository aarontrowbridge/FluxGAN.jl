using FluxGAN
using BSON: @load
using Flux

# FIXME: loading errs here
@load "models/cifar10_goodfellow_MLP_n_25.bson" model

layout = (x=4, y=3)

output_dir = "images/CIFAR10"

info = ["cifar10", "n", "25"]

image_grid(model, model.hparams.img_size, output_dir,
           layout=layout,
           file_info=info)