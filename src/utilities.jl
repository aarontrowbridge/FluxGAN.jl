using Images

function color_image(tensor::Array{T, 3}) where {T<:Number}
    tensor = permutedims(tensor, (3, 1, 2))
    return reverse(colorview(RGB, tensor), dims=2)
end

