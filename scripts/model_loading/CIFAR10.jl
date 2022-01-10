using GAN
using BSON: @load
using Flux
using CairoMakie
using CUDA

@load "models/cifar10_goodfellow_MLP.bson" model

layout = (x=4, y=3)

fig = Figure(resolution=(layout.x * 150, layout.y * 150))
for i = 1:layout.y, j = 1:layout.x
    fake = reshape(model.G(randn(Float32, model.hparams.latent_dim)), (32, 32, 3))
    ax, = image(fig[i,j], color_image(@. (fake + 1f0) / 2f0))
    hidedecorations!(ax)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, ticks=false)
end
save("fake_images/CIFAR10/goodfellow_mlp_loaded.png", fig)

