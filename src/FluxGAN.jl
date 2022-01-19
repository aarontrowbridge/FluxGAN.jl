module FluxGAN 

include("models.jl")
using .Models

export GANHyperParams 
export GAN 

include("image_utilities.jl")
using .ImageUtilities

export save_image_grid
export image_grid_figure
export image_grid_tensor
export image_grid
export color_image

include("train.jl")
using .Train

export train!


end
