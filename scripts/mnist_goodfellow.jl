using GAN
using Flux
using BSON: @save
using MLDatasets
using CUDA
CUDA.allowscalar(true)

const device = eval(Symbol(ARGS[1]))
const iterations = parse(Int, ARGS[2]) 
const skip = parse(Int, ARGS[3]) 

images = MNIST.traintensor(Float32)

const img_size = size(images)[1:end-1]
const img_num  = size(images)[end]
const img_dim  = *(img_size...) 

# centering pixels, mapping to [-1,1]; and flattening image tensors
images = reshape((@. 2f0 * images - 1f0), (img_dim, img_num)) 

generator = Chain(
    Dense(100, 1200, relu),
    Dense(1200, 1200, relu),
    Dense(1200, img_dim, tanh_fast)
)

discriminator = Chain(
    Maxout(() -> Dense(img_dim, 240), 5),
    Maxout(() -> Dense(240, 240), 5),
    Dense(240, 1)
)

# learning rate
const η = 0.0002f0

model = GANModel(generator, discriminator, η_gen=η, η_dscr=η) 

train!(model, images, skip=skip, device=device, iterations=iterations)

# uncomment below to save model when done training
# @save "models/mnist_goodfellow_MLP_test.bson" model

println("saving generated images...")
println()

using CairoMakie

layout = (x=5, y=4)
fig = Figure(resolution=(layout.x * 100, layout.y * 100))
for i = 1:layout.y, j = 1:layout.x
    img = reshape(model.G(randn(Float32, hparams.latent_dim) |> device) |> cpu, img_size)
    ax, = image(fig[i,j], @. (img + 1f0) / 2f0)
    hidedecorations!(ax)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, ticks=false)
end
save("fake_images/MNIST/gen_image_grid_n_$(iterations)_$(device).png", fig)

println("finished!")
println()
