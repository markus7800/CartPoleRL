include("../CartPole/all.jl")
include("../SemiGradientSARSA.jl")

using Random

cartpole = CartPole(xlims=(-1.,1.), mc=10., r=1., mp=1., theta=π/2)
plot_cartpole(cartpole)

# works for balancing
Random.seed!(1)
model = Chain(Dense(4+1, 10, sigmoid), Dense(10, 1))

agent = SemiGradientSARSA(model=model, α=0.1, ϵ=0.1, γ=0.9)

Random.seed!(1)
snapshots = learn_balance!(agent, cartpole, 5000, 2500, snapshot=250)


e, m, t = snapshots[5]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate_animate(cartpole, 15., force(agent), agent.n_inter, quit_if_done=true, ylab="1000 episodes", fps=agent.fps)
gif(anim, "single_balance/1000.gif")

e, m, t = snapshots[9]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate_animate(cartpole, 15., force(agent), agent.n_inter, quit_if_done=true, ylab="2000 episodes", fps=agent.fps)
gif(anim, "single_balance/2000.gif")

e, m, t = snapshots[13]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate_animate(cartpole, 30., force(agent), agent.n_inter, quit_if_done=true, ylab="3000 episodes", fps=agent.fps)
gif(anim, "single_balance/3000.gif")

e, m, t = snapshots[end]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate_animate(cartpole, 30., force(agent), agent.n_inter, quit_if_done=true, ylab="4375 episodes", fps=agent.fps)
gif(anim, "single_balance/final.gif")
