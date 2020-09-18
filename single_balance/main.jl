include("../CartPole.jl")
include("../SemiGradientSARSA.jl")

using Random

# works for balancing
Random.seed!(1)
model = Chain(Dense(4+1, 10, sigmoid), Dense(10, 1))

cartpole = CartPole(0, (-1.,1.), 0, 10., 1., 1., π/2, 0.)

agent = SemiGradientSARSA(model, 0.1, 0.1, 0.9, [-1., 1.], 100)

Random.seed!(1)
snapshots = learn_balance!(agent, cartpole, 5000, 2500, snapshot=250)

# reset_balance!(cartpole)
# anim = simulate(cartpole, 15., force(agent), agent.n_inter)
# gif(anim, "temp.gif")

# plot(map(s -> s[1], snapshots), map(s -> s[3], snapshots) .+ 1, yaxis=:log)

e, m, t = snapshots[5]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate(cartpole, 15., force(agent), agent.n_inter, quit_if_done=true, ylab="1000 episodes")
gif(anim, "single_balance/1000.gif")

e, m, t = snapshots[9]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate(cartpole, 15., force(agent), agent.n_inter, quit_if_done=true, ylab="2000 episodes")
gif(anim, "single_balance/2000.gif")

e, m, t = snapshots[13]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate(cartpole, 30., force(agent), agent.n_inter, quit_if_done=true, ylab="3000 episodes")
gif(anim, "single_balance/3000.gif")

e, m, t = snapshots[end]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate(cartpole, 30., force(agent), agent.n_inter, quit_if_done=true, ylab="4375 episodes")
gif(anim, "single_balance/final.gif")
