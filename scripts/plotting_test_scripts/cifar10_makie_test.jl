using CairoMakie
using MLDatasets
using Images

train_tensor, train_labels = CIFAR10.traindata(Float32)

animal_classnames = [
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse"    
]

classnames = CIFAR10.classnames()

animal_indices = findall(lab -> classnames[lab + 1] in animal_classnames, train_labels)

animal_tensor = train_tensor[:,:,:,animal_indices]

layout = (3, 4)

fig = Figure(resolution=reverse(layout).*150);

function color_image(tensor::Array{T, 3}) where {T<:Number}
    tensor = permutedims(tensor, (3, 1, 2))
    return reverse(colorview(RGB, tensor), dims=2)
end

for i = 1:layout[1], j = 1:layout[2] 
    animal = animal_tensor[:,:,:,rand(1:size(animal_tensor)[end])]
    ax, = image(fig[i,j], color_image(animal))
    hidedecorations!(ax)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, ticks=false)
end

save("./test_animals.png", fig)