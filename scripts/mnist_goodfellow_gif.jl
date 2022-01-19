using FluxGAN
using MLDatasets
using Flux
using BSON: @save

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

# learning rate
const η = 0.0002f0

# minibatch size

model = GAN(generator, discriminator; 
    minibatch=m, 
    η_gen=η, 
    η_dscr=η, 
    latent_dim=d,
    img_size=img_size
) 

println("\nbeginning training!\n")

train!(model, images, iterations=n, skip=skip,
    gif=true,
    gif_filename="MNIST/mlp",
    gif_fps=5
)

println("\nfinished!\n")
