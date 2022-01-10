using GAN
using Flux
using BSON: @save
using MLDatasets
using ImageCore
using CUDA
CUDA.allowscalar(true)

const device = eval(Symbol(ARGS[1]))
const skip = parse(Int, ARGS[3]) 
const n = parse(Int, ARGS[2]) 


# dataset  
train_tensor, train_labels = CIFAR10.traindata(Float32)

animal_classnames = [
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse"    
]

classnames = CIFAR10.classnames()

# removing any images that aren't animals
animal_indices = findall(lab -> classnames[lab + 1] in animal_classnames, train_labels)
animal_tensor = train_tensor[:,:,:,animal_indices]

const img_size = size(animal_tensor)[1:end-1]
const img_num  = size(animal_tensor)[end]
const img_dim  = *(img_size...) 

# centering pixels and flattening images
animal_tensor = reshape((@. 2f0 * animal_tensor - 1f0), (img_dim, img_num))

const d = 100

generator = Chain(
    Dense(d, 8000, relu),
    Dense(8000, 8000, tanh_fast),
    x -> reshape(x, (10, 10, 80)),
    ConvTranspose((5, 5), 80 => 3, tanh_fast)
)

discriminator = Chain(
    Maxout(() -> Conv((8, 8), 3 => 32, pad=4), 2),
    MeanPool((4, 4), stride=(2, 2)),
)

# learning rate
const η = 0.0001f0

model = GANModel(generator, discriminator, η_gen=η, η_dscr=η)

train!(model, animal_tensor, skip=skip, device=device, iterations=n, latent_dim=d)

# uncomment below to save model after training
# @save "models/cifar10_goodfellow_MLP_test.bson" model

println("saving generated images...")

using CairoMakie

layout = (x=4, y=3)

fig = Figure(resolution=(layout.x * 150, layout.y * 150))
for i = 1:layout.y, j = 1:layout.x
    img = reshape(
        model.G(randn(Float32, model.hparams.latent_dim) |> device) |> cpu,
        img_size
    )
    ax, = image(fig[i,j], color_image(@. (img + 1f0) / 2f0))
    hidedecorations!(ax)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, ticks=false)
end
save("fake_images/CIFAR10/gen_image_grid_n_$(iterations)_$(device).png", fig)

println("finished!")
println()
