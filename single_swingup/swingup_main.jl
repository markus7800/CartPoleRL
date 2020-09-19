include("../CartPole/all.jl")
include("../solve_swingup_BFS.jl")
include("../SemiGradientSARSA.jl")


using Random

cartpole = CartPole(xlims=(-1.,1.), mc=10., r=1., mp=1., theta=-π/2)
plot_cartpole(cartpole)

FPS = 30
APS = 10
N_INTER = 100

# swing up
@time F, = brute_swingupBFS(cartpole, APS, FPS, 3, N_INTER)


# balance
Random.seed!(1)
model = Chain(Dense(4+1, 10, sigmoid), Dense(10, 1))

agent = SemiGradientSARSA(model=model, α=0.1, ϵ=0.1, γ=0.9, n_inter=N_INTER, fps=FPS)

Random.seed!(1)
snapshots = learn_balance!(agent, cartpole, 5000, 2500, snapshot=250)

function force(F::Vector{Float64}, aps::Int, fps::Int, agent::SemiGradientSARSA)
    @assert fps % aps == 0
    na = fps ÷ aps
    return function force_f(cartp::CartPole, i, t)
        a = Int(i ÷ na + 1)
        if a ≤ length(F)
            return F[a] * cartp.mc * 10 # swingup
        else
            A = greedy_action(agent, state(cartp))
            return A * cartp.mc * 10
        end
    end
end


reset_swingup!(cartpole)
anim = simulate_animate(cartpole, 30, force(F, APS, FPS, agent), N_INTER)
gif(anim, "single_swingup/swingup.gif")
gif(anim, "temp.gif")
