include("../CartPole/all.jl")


using Random

cartpole = CartPole(xlims=(-1.,1.), mc=10., r=1., mp=1., theta=-π/2)
plot_cartpole(cartpole)

# works for balancing
Random.seed!(1)
model = Chain(Dense(4+1, 10, sigmoid), Dense(10, 10, sigmoid), Dense(10, 1))

agent = SemiGradientSARSA(model=model, α=.1, ϵ=0.2, γ=1.0)

Random.seed!(1)
snapshots = learn_swingup!(agent, cartpole, 5000, 2500, snapshot=250)


reset_swingup!(cartpole)
anim = simulate(cartpole, 15., force(agent), agent.n_inter, quit_if_done=true, ylab="episodes", fps=agent.fps)
gif(anim, "single_swingup/temp.gif")


e, m, t = snapshots[18]
agent.Q̂ = m
reset_swingup!(cartpole)
anim = simulate(cartpole, 15., force(agent), agent.n_inter, quit_if_done=true, ylab="$e episodes", fps=agent.fps)
gif(anim, "single_swingup/temp.gif")

e, m, t = snapshots[9]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate(cartpole, 15., force(agent), agent.n_inter, quit_if_done=true, ylab="2000 episodes", fps=agent.fps)
gif(anim, "single_balance/2000.gif")

e, m, t = snapshots[13]
agent.Q̂ = m
reset_balance!(cartpole)
anim = simulate(cartpole, 30., force(agent), agent.n_inter, quit_if_done=true, ylab="3000 episodes", fps=agent.fps)
gif(anim, "single_balance/3000.gif")

e, m, t = snapshots[end]
agent.Q̂ = m
reset_swingup!(cartpole)
anim = simulate(cartpole, 30., force(agent), agent.n_inter, quit_if_done=true, ylab="4375 episodes", fps=agent.fps)
gif(anim, "single_swingup/final.gif")



function force_f(cartpole, t)
    if t < 0.4
        1. * cartpole.mc * 10
    elseif t < 1.
        -1. * cartpole.mc * 10
    else
        1. * cartpole.mc * 10
    end
end

reset_swingup!(cartpole)
anim = simulate(cartpole, 15., force_f, agent.n_inter, quit_if_done=true)
gif(anim, "single_swingup/temp.gif")
