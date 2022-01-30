using WGLMakie
WGLMakie.activate!()

x₀ = 0.0
Δx = 0.1

fps = 50
nframes = 500

xs = Observable([x₀])
ys = Observable([sin(x₀)])

fig, ax, = lines(ys, figure=(resolution=(500,400),))
limits!(ax, 0, nframes, -1.5, 1.5)

xs[]

@time for i = 1:nframes
    x = xs[][i] + Δx 
    xs[] = push!(xs[], x)
    ys[] = push!(ys[], sin(x))
    sleep(1 / fps)
end

xs[]

fig