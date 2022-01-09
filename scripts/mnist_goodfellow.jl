println("loading packages...")

using GAN
using Flux
using BSON: @save
using MLDatasets
using CUDA
CUDA.allowscalar(true)

println("packages loaded!")
println()

# device
const device = eval(Symbol(ARGS[1]))

println("loading MNIST images...")

# dataset  
images = MNIST.traintensor(Float32)

# image dimensions 
const img_size = size(images)[1:end-1]
const img_num  = size(images)[end]
const img_dim  = *(img_size...) 

images = reshape((@. 2f0 * images - 1f0), (img_dim, img_num)) 

println("images are loaded!")
println()
# println("image size = ", img_size)
# println("image dim  = ", img_dim)
# println("image num  = ", img_num)
# println()

println("loading generator...")

generator = Chain(
    Dense(100, 1200, relu),
    Dense(1200, 1200, relu),
    Dense(1200, img_dim, tanh_fast)
)

println("generator loaded!")
println()

println("loading discriminator...")

discriminator = Chain(
    Maxout(() -> Dense(img_dim, 240), 5),
    Maxout(() -> Dense(240, 240), 5),
    Dense(240, 1)
)

println("discriminator loaded!")
println()

println("creating model...")

# training loops
const iterations = parse(Int, ARGS[2]) 

const η = 0.002f0

hparams = GANHyperParams(iterations = iterations, η_gen = η, η_dscr = η)
model = GANModel(generator, discriminator, hparams) |> device 

println("model created!")
println()

println("training model...")
println()

const skip = parse(Int, ARGS[3]) 
train!(model, images, skip=skip, device=device)

println("model trained!")
println()

println()
println("saving model...")

@save "models/mnist_goodfellow_MLP_test.bson" model

println("saving generated images...")

using CairoMakie

dims = (x=5, y=4)
fig = Figure(resolution=(dims.x * 100, dims.y * 100))
for i = 1:dims.y, j = 1:dims.x
    img = reshape(model.G(randn(Float32, hparams.latent_dim) |> device) |> cpu, img_size)
    ax, = image(fig[i,j], @. (img + 1f0) / 2f0)
    hidedecorations!(ax)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, ticks=false)
end
save("fake_images/MNIST/gen_image_grid_n_$(iterations)_$(device).png", fig)

println("finished!")
println()
