using Revise
using GAN

using Flux, Distributions

using GLMakie
using MLDatasets 

images = Flux.unstack(MNIST.traintensor(Float32), 3)

# generator layer structure (nodes, activation)
G = [(50,  sigmoid_fast),
     (75,  sigmoid_fast),
     (100, sigmoid_fast),
     (200, sigmoid_fast)]


# generator layer structure (nodes, activation)
D = [(200, sigmoid_fast),
     (100, sigmoid_fast),
     (50,  sigmoid_fast),
     (10,  sigmoid_fast)] 

hyper_params = GANHyperParamsMLP(
    size(images[1]), 
    G, D, 
    NoiseDist=Normal()
)

model = GANmodel(hyper_params)

train!(model, images, n=5000, k=10)

# image_diff = model.G(rand(model.hps.d)) - model.G(rand(model.hps.d))
# image(reshape(image_diff, model.hps.data_size))

image(reshape(model.G(rand(model.hps.d)), model.hps.data_size))