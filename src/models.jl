using Flux
using Flux.Losses: logitbinarycrossentropy 
using Functors: @functor
using Distributions
using Statistics
using Random
using Parameters

@with_kw struct GANHyperParams 
    latent_dim::Int = 100            
    minibatch::Int  = 50
    iterations::Int = 1000
    dscr_loops::Int = 1
    η_dscr::Float32 = 0.002
    η_gen::Float32  = 0.002
end

mutable struct GANModel
    G::Chain
    D::Chain
    hparams::GANHyperParams
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

function train_discriminator!(model, opt, xs, zs)
    ω = Flux.params(model.D)
    loss, ∇ = Flux.pullback(ω) do 
        discriminator_loss(model.D(xs), model.D(model.G(zs)))
    end 
    Flux.update!(opt, ω, ∇(1f0))
    return loss
end

function train_generator!(model, opt, zs)
    θ = Flux.params(model.G)
    loss, ∇ = Flux.pullback(θ) do 
        generator_loss(model.D(model.G(zs)))
    end 
    Flux.update!(opt, θ, ∇(1f0))
    return loss
end


# model training function

function train!(model::GANModel, train_tensor::AbstractArray;
                opt=OADAM,
                device=cpu,
                verbose=true, 
                skip=50)

    model = model |> device

    train_tensor_size = size(train_tensor)
    
    d = model.hparams.latent_dim
    m = model.hparams.minibatch

    G_opt = opt(model.hparams.η_gen)     
    D_opt = opt(model.hparams.η_dscr)

    t0 = time_ns() 

    gen_loss = 0 
    dscr_loss = 0 

    for i = 1:model.hparams.iterations

        if verbose && i % skip == 0
            t = time_ns()
            println("iteration: $i of ", model.hparams.iterations)
            println("avg seconds per loop   = ", Float32((t - t0) / skip)*1f-9) 
            println("avg generator loss     = ", Float32(gen_loss / skip))
            println("avg discriminator loss = ", Float32(dscr_loss / skip))
            println()
            gen_loss = 0
            dscr_loss = 0
            t0 = time_ns()
        end

        for _ = 1:model.hparams.dscr_loops 
            x_tensor = train_tensor[:,rand(1:train_tensor_size[end], m)] |> device
            z_tensor = randn!(similar(train_tensor, (d, m))) |> device
            dscr_loss += train_discriminator!(model, D_opt, x_tensor, z_tensor)
        end

        z_tensor = randn!(similar(train_tensor, (d, m))) |> device
        gen_loss += train_generator!(model, G_opt, z_tensor)
    end
end



