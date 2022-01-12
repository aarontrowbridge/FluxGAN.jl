module Train

export train!

using FluxGAN: GAN

using Flux
using Flux.Losses: logitbinarycrossentropy 
using EllipsisNotation
using CUDA

######################
# training functions #
######################

# main training function - attempts to train on GPU

function train!(model::GAN, train_tensor::AbstractArray; kws...)
    if has_cuda_gpu()
        println("training GAN with GPU...")
        train_gpu!(model, train_tensor; kws...)
    else
        println("no GPU available :( training GAN with CPU...")
        train_cpu!(model, train_tensor; kws...)
    end
end

# gpu training function

function train_gpu!(model::GAN, train_tensor::AbstractArray; 
                    iterations=1000, verbose=true, skip=50)

    d = model.hparams.latent_dim
    m = model.hparams.minibatch

    gen_loss = 0 
    dscr_loss = 0 

    t0 = time_ns() 

    model.G = model.G |> gpu 
    model.D = model.D |> gpu 
    for i = 1:iterations
        if verbose && i % skip == 0
            training_message(i, iterations, t0, gen_loss, dscr_loss, skip)
            gen_loss = 0
            dscr_loss = 0
            t0 = time_ns()
        end
        for _ = 1:model.hparams.dscr_loops 
            x_tensor = train_tensor[.., rand(1:size(train_tensor)[end], m)] |> gpu
            z_tensor = CUDA.randn(Float32, d, m) 
            dscr_loss += train_discriminator!(model, x_tensor, z_tensor)
        end
        z_tensor = CUDA.randn(Float32, d, m) 
        gen_loss += train_generator!(model, z_tensor)
    end
    model.G = model.G |> cpu
    model.D = model.D |> cpu
end

# cpu training function

function train_cpu!(model::GAN, train_tensor::AbstractArray; 
                    iterations=1000, verbose=true, skip=50)

    d = model.hparams.latent_dim
    m = model.hparams.minibatch

    gen_loss = 0 
    dscr_loss = 0 

    t0 = time_ns() 

    for i = 1:iterations
        if verbose && i % skip == 0
            training_message(i, iterations, t0, gen_loss, dscr_loss, skip)
            gen_loss = 0
            dscr_loss = 0
            t0 = time_ns()
        end
        for _ = 1:model.hparams.dscr_loops 
            x_tensor = train_tensor[.., rand(1:size(train_tensor)[end], m)]
            z_tensor = randn(Float32, d, m) 
            dscr_loss += train_discriminator!(model, x_tensor, z_tensor)
        end
        z_tensor = randn(Float32, d, m) 
        gen_loss += train_generator!(model, z_tensor)
    end
end

# discriminator train function

function train_discriminator!(model::GAN, xs, zs)
    ω = Flux.params(model.D)
    loss, ∇ = Flux.pullback(ω) do 
        discriminator_loss(model.D(xs), model.D(model.G(zs)))
    end 
    ∇ = ∇(1f0)
    Flux.update!(model.D_opt, ω, ∇)
    return loss
end

# generator train function

function train_generator!(model::GAN, zs)
    θ = Flux.params(model.G)
    loss, ∇ = Flux.pullback(θ) do 
        generator_loss(model.D(model.G(zs)))
    end 
    ∇ = ∇(1f0)
    Flux.update!(model.G_opt, θ, ∇)
    return loss
end

# training message function

# TODO: add plotting functionality
#       * mean training loss
#       * maybe error bars

function training_message(i, iterations, t0, gen_loss, dscr_loss, skip)
    t = time_ns()
    println("\niteration: $i of ", iterations)
    println("avg seconds per loop   = ", Float32((t - t0) / skip)*1f-9) 
    println("avg generator loss     = ", Float32(gen_loss / skip))
    println("avg discriminator loss = ", Float32(dscr_loss / skip))
end

##################
# loss functions #
##################
 
function discriminator_loss(real_outputs, fake_outputs)
    real_loss = logitbinarycrossentropy(real_outputs, 1) 
    fake_loss = logitbinarycrossentropy(fake_outputs, 0) 
    return real_loss + fake_loss
end

generator_loss(fake_outputs) = logitbinarycrossentropy(fake_outputs, 1)

end