using FluxGAN
using MLDatasets
using Flux
using CairoMakie
using LaTeXStrings

const n = parse(Int, ARGS[1]) 
const m = parse(Int, ARGS[2])
const skip = parse(Int, ARGS[3]) 

images = MNIST.traintensor(Float32)

const img_size = size(images)[1:end-1]
const img_num  = size(images)[end]
const img_dim  = *(img_size...) 

# centering pixels, mapping to [-1,1]; and flattening image tensors
images = reshape((@. 2f0 * images - 1f0), (img_dim, img_num)) 

const d = 100

generator = Chain(
    Dense(d, 1200, relu),
    Dense(1200, 1200, relu),
    Dense(1200, img_dim, tanh_fast)
)

discriminator = Chain(
    Maxout(() -> Dense(img_dim, 240), 5),
    Maxout(() -> Dense(240, 240), 5),
    Dense(240, 1)
)

# minibatch size

ks = [1, 5, 20, 40]
titles = [latexstring("\$k = \$$k") for k in ks ]

fig = Figure()
ax1 = Axis(fig[1,1], xlabel="iteration", ylabel="loss", title=titles[1])
ax2 = Axis(fig[1,2], xlabel="iteration", ylabel="loss", title=titles[2])
ax3 = Axis(fig[2,1], xlabel="iteration", ylabel="loss", title=titles[3])
ax4 = Axis(fig[2,2], xlabel="iteration", ylabel="loss", title=titles[4])
axs = [ax1, ax2, ax3, ax4]

ts = collect(skip:skip:n) 

maxs = []

for (i, k) in enumerate(ks) 
    model = GAN(deepcopy(generator), deepcopy(discriminator); 
        minibatch=m, 
        latent_dim=d,
        img_size=img_size,
        dscr_loops=k
    ) 
    gen_losses, dscr_losses = train!(model, images; 
                                     iterations=n, 
                                     skip=skip, 
                                     return_losses=true)
    push!(maxs, maximum([gen_losses; dscr_losses]))
    lines!(axs[i], ts, gen_losses; label=L"G", color=:green)
    lines!(axs[i], ts, dscr_losses; label=L"D", color=:purple)
    axislegend(axs[i])
end

max = maximum(maxs)

for ax in axs
    ylims!(ax, 0, max)
end

save("images/loss_plots/loss_for_ks_n_$n.png", fig)