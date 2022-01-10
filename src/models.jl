using Flux
using Flux.Losses: logitbinarycrossentropy 
using Flux.Optimise: AbstractOptimiser
using Functors: @functor
using Distributions
using Statistics
using Random
using Parameters
using EllipsisNotation
using CUDA

@with_kw struct GANHyperParams 
    latent_dim::Int = 100            
    dscr_loops::Int = 1
    minibatch::Int  = 50
    η_gen::Float32  = 0.0001
    η_dscr::Float32 = 0.0001
end

mutable struct GANModel
    G::Chain
    D::Chain
    G_opt::AbstractOptimiser
    D_opt::AbstractOptimiser
    hparams::GANHyperParams
    
    function GANModel(G::Chain, D::Chain; kws...) 
        hparams = GANHyperParams(; kws...)
        G_opt = OADAM(hparams.η_gen)
        D_opt = OADAM(hparams.η_dscr)
        return new(G, D, G_opt, D_opt, hparams)
    end

    function GANModel(G::Chain, D::Chain, opt::AbstractOptimiser; kws...) 
        hparams = GANHyperParams(; kws...)
        G_opt = opt(hparams.η_gen)
        D_opt = opt(hparams.η_dscr)
        return new(G, D, G_opt, D_opt, hparams)
    end
end

@functor GANModel

# loss functions

function discriminator_loss(real_outputs, fake_outputs)
    real_loss = logitbinarycrossentropy(real_outputs, 1) 
    fake_loss = logitbinarycrossentropy(fake_outputs, 0) 
    return real_loss + fake_loss
end

generator_loss(fake_outputs) = logitbinarycrossentropy(fake_outputs, 1)


# training functions

function train_discriminator!(model::GANModel, xs, zs)
    ω = Flux.params(model.D)
    loss, ∇ = Flux.pullback(ω) do 
        discriminator_loss(model.D(xs), model.D(model.G(zs)))
    end 
    ∇ = ∇(1f0)
    Flux.update!(model.D_opt, ω, ∇)
    return loss
end

function train_generator!(model::GANModel, zs)
    θ = Flux.params(model.G)
    loss, ∇ = Flux.pullback(θ) do 
        generator_loss(model.D(model.G(zs)))
    end 
    ∇ = ∇(1f0)
    Flux.update!(model.G_opt, θ, ∇)
    return loss
end

# model training function

function train!(model::GANModel, train_tensor;
                iterations=1000,
                device=cpu,
                verbose=true, 
                skip=50)

    model.G = model.G |> device
    model.D = model.D |> device

    d = model.hparams.latent_dim
    m = model.hparams.minibatch

    t0 = time_ns() 

    gen_loss = 0 
    dscr_loss = 0 

    for i = 1:iterations

        if verbose && i % skip == 0
            t = time_ns()
            println("iteration: $i of ", iterations)
            println("avg seconds per loop   = ", Float32((t - t0) / skip)*1f-9) 
            println("avg generator loss     = ", Float32(gen_loss / skip))
            println("avg discriminator loss = ", Float32(dscr_loss / skip))
            println()
            gen_loss = 0
            dscr_loss = 0
            t0 = time_ns()
        end

        for _ = 1:model.hparams.dscr_loops 
            x_tensor = train_tensor[.., rand(1:size(train_tensor)[end], m)] |> device
            z_tensor = randn!(similar(train_tensor, (d, m))) |> device
            dscr_loss += train_discriminator!(model, x_tensor, z_tensor)
        end

        z_tensor = randn!(similar(train_tensor, (d, m))) |> device
        gen_loss += train_generator!(model, z_tensor)
    end

    model.G = model.G |> cpu
    model.D = model.D |> cpu
end



