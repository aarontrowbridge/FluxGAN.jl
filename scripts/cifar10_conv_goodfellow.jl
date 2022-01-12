using Flux
using FluxGAN
using BSON: @save
using MLDatasets

const n = parse(Int, ARGS[1]) 
const m = parse(Int, ARGS[2])
const skip = parse(Int, ARGS[3]) 


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

# centering pixels
animal_tensor = @. 2f0 * animal_tensor - 1f0

const d = 100

generator = Chain(
    Dense(d, 8000, relu),
    Dense(8000, 8000, tanh_fast),
    x -> reshape(x, 10, 10, 80, size(x)[end]),
    ConvTranspose((5, 5), 80 => 3, tanh_fast; stride=(3,3))
)

discriminator = Chain(
    Maxout(() -> Conv((8, 8), 3 => 32; pad=4), 2),
    MeanPool((4, 4); stride=(2, 2)),
    Maxout(() -> Conv((8, 8), 32 => 32, pad=3), 2),
    MeanPool((4, 4); stride=(2, 2)),
    Maxout(() -> Conv((5, 5), 32 => 192, pad=3), 2),
    MeanPool((2, 2); stride=(2, 2)),
    x -> reshape(x, 3072, size(x)[end]),
    Maxout(() -> Dense(3072, 500), 5),
    Dense(500, 1)
)

# learning rate
const η = 0.0001f0

model = GAN(generator, discriminator; 
    η_gen=η, 
    η_dscr=η, 
    minibatch=m,
    latent_dim=d,
    img_size=img_size
)

train!(model, animal_tensor, skip=skip, iterations=n)

# uncomment below to save model after training
# FIXME: saving errs here

# println("\nsaving model...")
# @save "models/cifar10_goodfellow_conv_n_$(n).bson" model
# println("model saved!")

println("\nsaving generated images...")

layout = (x=5, y=5)

output_dir = "images/CIFAR10"

info = ["animals", "conv", "n", n, "m", m]

image_grid(model, img_size, output_dir; 
    layout=layout, 
    file_info=info,
    img_res=150
)

println("finished!\n")
