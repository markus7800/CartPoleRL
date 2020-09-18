include("CartPole.jl")
include("SemiGradientSARSA.jl")

using Random

Random.seed!(1)
model = Chain(Dense(4+1, 10, sigmoid), Dense(10, 1))

cartpole = CartPole(0, (-2.,2.), 0, 10., 1., 1., π/2, 0.)

agent = SemiGradientSARSA(model, 0.01, 0.05, 0.9, [-1., 1.])

Random.seed!(1)
learn_balance!(agent, cartpole, 1000)

reset_balance!(cartpole)
anim = simulate(cartpole, 15., force(agent), 100)
gif(anim, "temp.gif")



# works for balancing
Random.seed!(1)
model = Chain(Dense(4+1, 10, sigmoid), Dense(10, 1))

cartpole = CartPole(0, (-1.,1.), 0, 10., 1., 1., π/2, 0.)

agent = SemiGradientSARSA(model, 0.1, 0.1, 0.9, [-1., 1.])

Random.seed!(1)
learn_balance!(agent, cartpole, 1000)

reset_balance!(cartpole)
anim = simulate(cartpole, 15., force(agent), 100)
gif(anim, "temp.gif")
