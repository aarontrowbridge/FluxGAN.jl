println("loading packages...")

using GAN
using Flux
using BSON: @save
using MLDatasets
using ImageCore
using CUDA
CUDA.allowscalar(true)

println("packages loaded!")
println()

# device
const device = eval(Symbol(ARGS[1]))

println("loading CIFAR10 animal images...")

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

animal_indices = findall(lab -> classnames[lab + 1] in animal_classnames, train_labels)

animal_tensor = train_tensor[:,:,:,animal_indices]

# image dimensions 
const img_size = size(animal_tensor)[1:end-1]
const img_num  = size(animal_tensor)[end]
const img_dim  = *(img_size...) 

animal_tensor = reshape((@. 2f0 * animal_tensor - 1f0), (img_dim, img_num))

println("images are loaded!")
println()
println("image size = ", img_size)
println("image dim  = ", img_dim)
println("image num  = ", img_num)
println()

println("loading generator...")

generator = Chain(
    Dense(100, 8000, relu),
    Dense(8000, 8000, relu),
    Dense(8000, img_dim, sigmoid_fast)
)

println("generator loaded!")
println()

println("loading discriminator...")

discriminator = Chain(
    Maxout(() -> Dense(img_dim, 1600), 5),
    Maxout(() -> Dense(1600, 1600), 5),
    Dense(1600, 1)
)

println("discriminator loaded!")
println()

println("creating model...")

# training loops
const iterations = parse(Int, ARGS[2]) 

const η = 0.002f0

hparams = GANHyperParams(
    iterations = iterations, 
    η_gen = η, 
    η_dscr = η,
    minibatch=50
)
model = GANModel(generator, discriminator, hparams)
println("model created!")
println()


println("training model...")
println()

const skip = parse(Int, ARGS[3]) 
train!(model, animal_tensor, skip=skip, device=device)

println("model trained!")
println()

println()
println("saving model...")

@save "models/cifar10_goodfellow_MLP_test.bson" model

println("saving generated images...")

using CairoMakie

layout = (x=4, y=3)

fig = Figure(resolution=(layout.x * 150, layout.y * 150))
for i = 1:layout.y, j = 1:layout.x
    img = reshape(model.G(randn(Float32, hparams.latent_dim)), img_size)
    ax, = image(fig[i,j], CIFAR10.convert2image(@. (img + 1f0) / 2f0))
    hidedecorations!(ax)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, ticks=false)
end
save("fake_images/CIFAR10/gen_image_grid_n_$(iterations)_$(device).png", fig)

println("finished!")
println()
