module Utilities

export color_image
export image_grid

using FluxGAN: GAN

using CairoMakie
using Images
using Dates
using EllipsisNotation

#####################
# utility functions #
#####################

# return color RGB image from image tensor

function color_image(imgtensor::Array)
    @assert length(size(imgtensor)) ∈ [2, 3] "Image tensor dimension ∉ [2, 3]!"
    if length(size(imgtensor)) == 3
        imgtensor = permutedims(imgtensor, (3, 1, 2))
        img = Matrix(colorview(RGB, imgtensor))
    else
        img = Matrix(colorview(Gray, imgtensor)) 
    end
    return img
end

# return image grid from images tensor 

function image_grid(imgs::Array, layout::NamedTuple{(:x, :y), Tuple{Int,Int}};
                    img_res = 150, hflip=true)

    @assert size(imgs)[end] == layout.x*layout.y "Image number ≠ grid size!"

    imgs = [imgs[.., i] for i in 1:size(imgs)[end]]
    imgs = color_image.(imgs)

    if hflip imgs = map(img -> reverse(img, dims=2), imgs) end

    fig = Figure(resolution=(layout.x * img_res, layout.y * img_res))
    for i = 1:layout.y, j = 1:layout.x
        ax, = image(fig[i,j], imgs[i + (layout.y) * (j - 1)])
        hidedecorations!(ax)
        hidexdecorations!(ax, ticks=false)
        hideydecorations!(ax, ticks=false)
    end
    return fig
end

# save image grid plot from model 

function image_grid(model::GAN, output_dir::String;
                    layout = (x=5, y=4),
                    img_res = 150,
                    uncenter = true,
                    date = false,
                    file_info = [])

    @assert model.hparams.img_size ≠ undef "Image size not defined!"
    imgs = model.G(randn(Float32, model.hparams.latent_dim, layout.x * layout.y))
    imgs = reshape(imgs, model.hparams.img_size..., :) 
    if uncenter imgs = @. (imgs + 1f0) / 2f0 end

    fig = image_grid(imgs, layout, img_res=img_res)

    if file_info == []
        info = ""
    else
        info = mapreduce(tag -> string(tag) * "_", *, file_info) 
    end

    file = info * "grid_$(layout.x)_$(layout.y)" * 
           (date ? "_" * string(today()) : "") * ".png"

    save(output_dir * "/" * file, fig)
end

end
