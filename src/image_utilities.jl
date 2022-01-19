module ImageUtilities

export save_image_grid
export image_grid_figure
export image_grid_tensor
export image_grid
export color_image

using FluxGAN: GAN

using Flux: cpu
using CairoMakie
using Images
using FileIO
using Dates
using EllipsisNotation
using Random

###########################
# image utility functions #
###########################

# save image grid figure from model 
#
function save_image_grid(model::GAN, output_dir::String;
                    layout=(5, 5),
                    date=false,
                    file_info=[],
                    img_res=150,
                    kws...)
    if file_info == []
        info = ""
    else
        info = mapreduce(tag -> string(tag) * "_", *, file_info) 
    end
    file = info * "grid_$(layout[1])_$(layout[2])" * 
           (date ? "_" * string(today()) : "") 
    imgs = image_grid_tensor(model, layout.x * layout.y; kws...)
    fig = image_grid_figure(imgs, layout; img_res=img_res)
    save(output_dir * "/" * file * ".png", fig)
end

# return color images tensor from model
#
function image_grid_tensor(model::GAN, n::Int;
                           uncenter=true,
                           seed=false)
    @assert model.hparams.img_size ≠ undef "Image size not defined!"
    if seed Random.seed!(69) end
    imgs = cpu(model.G)(randn(Float32, model.hparams.latent_dim, n))
    if uncenter imgs = @. (imgs + 1f0) / 2f0 end
    imgs = reshape(imgs, model.hparams.img_size..., :) 
    color_imgs = Array{Color}(undef, size(imgs)[1:2]..., size(imgs)[end])
    for i = 1:size(imgs)[end]
        color_imgs[:,:,i] = color_image(imgs[..,i])
    end
    color_imgs
end

# convert images tensor into an image grid
#
function image_grid(imgs::Array{Color, 3}, layout::Tuple{Int,Int})
    h, w, n = size(imgs)
    rows, cols = layout
    @assert rows * cols == n "Number of images ≠ grid size!"
    grid = Array{Color}(undef, h * rows, w * cols)
    for i = 1:rows, j = 1:cols
        img = imgs[:,:, i + rows*(j - 1)]
        img = reverse(img, dims=2)
        grid[((i - 1)*h + 1):i*h, ((j - 1)*w + 1):j*w] = img 
    end
    grid
end

function image_grid(model::GAN, layout::Tuple{Int,Int}; kws...)
    imgs = image_grid_tensor(model, *(layout...); kws...)
    image_grid(imgs, layout)
end

# return image grid makie figure from image grid tensor 
#
function image_grid_figure(imgs::Array, layout::Tuple{Int,Int};
                           img_res=150, hflip=true)
    n = size(imgs)[end] 
    rows, cols = layout
    @assert rows * cols == n "Number of images ≠ grid size!"

    imgs = [imgs[.., i] for i in 1:size(imgs)[end]]
    imgs = color_image.(imgs)

    if hflip imgs = map(img -> reverse(img, dims=2), imgs) end

    fig = Figure(resolution=(cols * img_res, rows * img_res))
    for i = 1:rows, j = 1:cols
        ax, = image(fig[i,j], imgs[i + rows * (j - 1)])
        hidedecorations!(ax)
        hidexdecorations!(ax, ticks=false)
        hideydecorations!(ax, ticks=false)
    end
    fig
end

# return color RGB image from image array
#
function color_image(img::Array)
    @assert length(size(img)) ∈ [2, 3] "Image tensor dimension ∉ [2, 3]!"
    if length(size(img)) == 3
        img = permutedims(img, (3, 1, 2))
        img = Matrix(colorview(RGB, img))
    else
        img = Matrix(colorview(Gray, img)) 
    end
    img
end


end