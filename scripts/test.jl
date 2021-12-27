using Revise
using GAN

using Flux, Distributions

using GLMakie
using MLDatasets

images = Flux.unstack(MNIST.traintensor(Float32), 3)
img_size = size(images[1])

# generator layer structure (nodes, activation)
G = [(50, sigmoid_fast),
    (75, sigmoid_fast),
    (100, sigmoid_fast),
    (200, sigmoid_fast)]


# generator layer structure (nodes, activation)
D = [(200, sigmoid_fast),
    (100, sigmoid_fast),
    (50, sigmoid_fast),
    (10, sigmoid_fast)]

hyper_params = GANHyperParamsMLP(
    img_size,
    G, D,
    d=10, # internal noise vector (representation) dimension
    NoiseDist = Normal()
)

model = GANmodel(hyper_params)

train!(model, images, n = 5000, k = 10)

gen_img = reshape(
    model.G(rand(model.hps.NoiseDist, model.hps.d)),
    model.hps.data_size
)

image(gen_img)