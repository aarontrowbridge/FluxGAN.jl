module Models

export GANHyperParams
export GAN

using Flux
using Flux.Optimise: AbstractOptimiser
using Parameters

#################
# model structs #
#################

# hyperparameters

@with_kw struct GANHyperParams 
    latent_dim::Int = 100            
    dscr_loops::Int = 1
    minibatch::Int  = 50
    η_gen::Float32  = 0.0001f0
    η_dscr::Float32 = 0.0001f0
    img_size::Tuple = undef
end

# simple GAN model (GAN)

abstract type AbstractGAN end

mutable struct GAN <: AbstractGAN
    G::Chain{T} where T
    D::Chain{H} where H
    G_opt::AbstractOptimiser
    D_opt::AbstractOptimiser
    hparams::GANHyperParams
end

function GAN(G::Chain{T}, D::Chain{H}; kws...) where {T, H} 
    hparams = GANHyperParams(; kws...)
    G_opt = OADAM(hparams.η_gen)
    D_opt = OADAM(hparams.η_dscr)
    return GAN(G, D, G_opt, D_opt, hparams)
end

function GAN(G::Chain{T}, D::Chain{H}, opt::AbstractOptimiser; kws...) where {T,H} 
    hparams = GANHyperParams(; kws...)
    G_opt = opt(hparams.η_gen)
    D_opt = opt(hparams.η_dscr)
    return GAN(G, D, G_opt, D_opt, hparams)
end

function GAN(G::Chain{T}, D::Chain{H}, 
             G_opt::AbstractOptimiser, D_opt::AbstractOptimiser; kws...) where {T,H}
    hparams = GANHyperParams(; kws...)
    G_opt = G_opt(hparams.η_gen)
    D_opt = D_opt(hparams.η_dscr)
    return GAN(G, D, G_opt, D_opt, hparams)
end

Flux.@functor GAN
# TODO: add conditional GAN model (CondGAN)
# TODO: add art GAN model (ArtGAN)

end