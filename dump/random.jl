include("../CartPole.jl")
include("../SemiGradientSARSA.jl")

using Random

cartpole = CartPole(0, (-1.,1.), 0, 100., 0.5, 1., π/2, 0.)

function force_f(cp::CartPole, t)
    A = rand([-1.,1.])
    return A * cp.mc * 10
end


Random.seed!(1)
reset_swingup!(cartpole)
anim = simulate(cartpole, 15., force_f, 100, quit_if_done=true, ylab="episodes", fps=30)
gif(anim, "temp.gif")
