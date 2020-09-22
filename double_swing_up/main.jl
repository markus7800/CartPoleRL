include("../CartPole/all.jl")
include("../solve_swingup_BFS.jl")
include("../SemiGradientSARSA.jl")


using Random

dpc = DoubleCartPole(xlims=(-4.,4.), mc=10., r_1=1., mp_1=1., theta_1=π/4, r_2=1., mp_2=1., theta_2=π/2)
plot_cartpole(dpc)

FPS = 30
APS = 5
N_INTER = 100

# swing up
@time solv, = brute_swingupBFS(dpc, APS, FPS, 5, N_INTER, max_depth=20)

reset_swingup!(dpc)
anim = simulate_animate(dpc, 3, time_force(solv.best, APS, FPS), N_INTER)
gif(anim, "temp.gif")

F_ = solv.best[1:10]
reset_swingup!(dpc)
r,t,cartp = simulate(dpc, 3, time_force(F_, APS, FPS), N_INTER, quit_if_no_force=true)

@time solv, = brute_swingupBFS(dpc, APS, FPS, 5, N_INTER, max_depth=20, reset=false)

F = vcat(F_, solv.best)

reset_swingup!(dpc)
anim = simulate_animate(dpc, 3, time_force(F, APS, FPS), N_INTER)
gif(anim, "temp.gif")

dpc = DoubleCartPole(xlims=(-2.,2.), mc=10., r_1=1., mp_1=1., theta_1=π/4, r_2=1., mp_2=1., theta_2=π/2)
F = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
    # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0]

BenchmarkTools.@btime simulate(dpc, 1, time_force(F, APS, FPS), N_INTER)

# balance
Random.seed!(1)
model = Chain(Dense(6+1, 25, sigmoid), Dense(25, 1))

agent = SemiGradientSARSA(model=model, α=0.001, ϵ=0.1, γ=0.9, n_inter=N_INTER, fps=FPS)

Random.seed!(1)
snapshots = learn_balance!(agent, dpc, 10000, snapshot=1000)

reset_balance!(dpc)
anim = simulate_animate(dpc, 3, force(agent), N_INTER)
gif(anim, "temp.gif")
