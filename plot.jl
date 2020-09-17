

using Plots
gui()
x = 0:0.01:2*pi
p = plot(x, sin.(x))
gui(p)
for i in 1:20
    display(plot(x, sin.(x .+ i / 10.0)))
end

using Interact, Plots
data = randn(100)
plt_obs = Observable{Any}(scatter(data))
data_obs = Observable{Any}(data)

map!(t -> scatter(t), plt_obs, data_obs)

ui = dom"div"( plt_obs )

for i in 1:10
    data_obs[] = randn(100)
    sleep(0.4)
end
