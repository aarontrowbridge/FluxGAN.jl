using Flux
using Flux.Optimise: AbstractOptimiser
using Distributions

abstract type GANHyperParams end

mutable struct GANHyperParamsMLP <: GANHyperParams

    # data size
    data_size::Tuple

    # MLP structure hyper parameters
    G_layers::Vector # generator layer structure
    D_layers::Vector # discriminator layer structure

    # internal representation hyper parameters
    d::Int                  # internal representation dimension
    NoiseDist::Distribution # noise distribution

    function GANHyperParamsMLP(data_size::Tuple,
                               G_layers::Vector,
                               D_layers::Vector;
                               d=100,
                               NoiseDist=Uniform())
        return new(data_size, G_layers, D_layers, d, NoiseDist)
    end
end

mutable struct GANmodel
    G::Chain
    D::Chain
    hps::GANHyperParams
end

function GANmodel(hps::GANHyperParamsMLP)

    # dimension of vectorized images
    imgdim = *(hps.data_size...)

    # constructing generator 
    # d-dimensional noise vector -> MLP -> vector ∈ ℝ^data_dim

    G = [Dense(hps.d, hps.G_layers[1][1], hps.G_layers[1][2])]
    for l = 1:length(hps.G_layers) - 1
        layer = Dense(hps.G_layers[l][1], 
                      hps.G_layers[l + 1][1], 
                      hps.G_layers[l + 1][2])
        push!(G, layer)
    end
    push!(G, Dense(hps.G_layers[end][1], imgdim, sigmoid_fast))

    G = Chain(G...)

    # constructing discriminator:
    # image -> vec -> MLP -> scaler ∈ [0, 1]

    D = [vec, Dense(imgdim, hps.D_layers[1][1], hps.D_layers[1][2])]
    for l = 1:length(hps.D_layers) - 1
        layer = Dense(hps.D_layers[l][1], 
                      hps.D_layers[l + 1][1], 
                      hps.D_layers[l + 1][2])
        push!(D, layer)
    end
    push!(D, Dense(hps.D_layers[end][1], 1, sigmoid_fast))

    D = Chain(D...)

    return GANmodel(G, D, hps)
end

function train!(model::GANmodel, data::Vector{Matrix{Float32}}; 
                opt=ADAM(), # optimization method 
                n=1000,     # number of iterations  
                k=1,        # loops per iteration to spend on discriminator 
                m=10)       # minibatch size

    # internal representation params
    d = model.hps.d
    NoiseDist = model.hps.NoiseDist

    for i in 1:n

        if i % 50 == 0
            println("iteration = $i")
        end

        # train the discriminator

        for _ in 1:k 
            xs = rand(data, m)
            zs = [rand(NoiseDist, d) for _ in 1:m]

            θ_discriminator = Flux.params(model.D)

            ∇_discriminator = gradient(θ_discriminator) do 
                -1 / m * sum(zip(xs, zs)) do (x, z)
                    log(model.D(x)[1]) + log(1 .- model.D(model.G(z))[1])
                end
            end

            Flux.update!(opt, θ_discriminator, ∇_discriminator)
        end

        # train the generator

        zs = [rand(NoiseDist, d) for _ in 1:m]

        θ_generator = Flux.params(model.G)

        ∇_generator = gradient(θ_generator) do 
            1 / m * sum(zs) do z
                log(1 - model.D(model.G(z))[1])
            end
        end

        Flux.update!(opt, θ_generator, ∇_generator)
    end
end



