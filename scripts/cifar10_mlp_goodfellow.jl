using Flux
using FluxGAN
using FileIO
using MLDatasets

const n = parse(Int, ARGS[1])    # iterations 
const m = parse(Int, ARGS[2])    # minibatch size 
const skip = parse(Int, ARGS[3]) # verbose frequency 

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
animals_tensor = train_tensor[:,:,:,animal_indices]

const img_size = size(animals_tensor)[1:end-1]
const img_num  = size(animals_tensor)[end]
const img_dim  = *(img_size...) 

# centering pixels and flattening images
animals_tensor = reshape((@. 2f0 * animals_tensor - 1f0), (img_dim, img_num))

const d = 100

generator = Chain(
    Dense(d, 8000, relu),
    Dense(8000, 8000, tanh_fast),
    Dense(8000, img_dim, tanh_fast)
)

discriminator = Chain(
    Maxout(() -> Dense(img_dim, 1600), 5),
    Maxout(() -> Dense(1600, 1600), 5),
    Dense(1600, 1)
)

# learning rate
const η = 0.0001f0

# minibatch size

model = GAN(generator, discriminator; 
    η_gen=η, 
    η_dscr=η, 
    latent_dim=d, 
    minibatch=m,
    img_size=img_size
)

train!(model, animals_tensor, skip=skip, iterations=n)

# uncomment below to save model after training

println("\nsaving model...")
save("models/cifar10_goodfellow_MLP_n_$n.jld2", Dict("model" => model)) 
println("saved!\n")

# println("plotting generated images...")

# layout = (5, 5)

# output_dir = "images/CIFAR10"

# info = ["animals", "MLP", "n", n, "m", m]

# save_image_grid(model, output_dir; 
#     layout=layout, 
#     file_info=info,
#     img_res=150
# )

println("finished!\n")
