include("../CartPole/all.jl")
include("../solve_swingup_BFS.jl")
include("../SemiGradientSARSA.jl")


using Random

dpc = DoubleCartPole(xlims=(-2.,2.), mc=10., r_1=1., mp_1=1., theta_1=π/4, r_2=1., mp_2=1., theta_2=π/2)
plot_cartpole(dpc)

FPS = 30
APS = 10
N_INTER = 100

# swing up
@time F, = brute_swingupBFS(dpc, APS, FPS, 3, N_INTER, max_depth=25)

# reset_swingup!(dpc)
# anim = simulate_animate(dpc, 3, time_force(G, APS, FPS), N_INTER)
# gif(anim, "temp.gif")
