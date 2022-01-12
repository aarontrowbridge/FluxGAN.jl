module FluxGAN 

include("models.jl")
using .Models

export GANHyperParams 
export GAN 

include("train.jl")
using .Train

export train!


include("utilities.jl")
using .Utilities

export color_image
export image_grid

end
