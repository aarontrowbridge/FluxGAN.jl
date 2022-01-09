using Plots
using MLDatasets

imgs = CIFAR10.traintensor(Float32)

layout = (3, 4)

plts = []
for _ = 1:layout[1]*layout[2] 
    img = imgs[:,:,:,rand(1:size(imgs)[end])]
    rgb_img = [RGB(img[i,j,:]...) for i = 1:size(img)[1], j = 1:size(img)[2]]
    push!(plts, plot(rgb_img'))
end

plot(plts..., layout=layout, showaxis=false, ticks=false, margin="1mm")