using Flux
using Flux.Optimise
using Printf

mutable struct SemiGradientSARSA
    Q̂
    α::Float32
    ϵ::Float64
    γ::Float32
    actions::Vector{Float32}
    n_inter::Int
    fps::Int

    function SemiGradientSARSA(;model, α, ϵ, γ, n_inter=100, fps=30)
        this = new()
        this.Q̂ = model
        this.α = α
        this.ϵ = ϵ
        this.γ = γ
        this.actions = [-1f0,1f0]
        this.n_inter = n_inter
        this.fps = fps
        return this
    end
end

function reset_balance!(cartp::CartPole)
    cartp.x = 0
    cartp.v = 0
    cartp.theta = π/2
    cartp.theta_dot = 0
end

function reset_swingup!(cartp::CartPole)
    cartp.x = 0
    cartp.v = 0
    cartp.theta = -π/2
    cartp.theta_dot = 0
end


function greedy_action(agent::SemiGradientSARSA, X::Vector{Float32})
    Qs = map(a -> agent.Q̂(vcat(X, a)), agent.actions)
    return agent.actions[argmax(Qs)]
end

function ϵ_greedy_action(agent::SemiGradientSARSA, X::Vector{Float32})
    if rand() ≤ agent.ϵ
        return rand(agent.actions)
    else
        return greedy_action(agent, X)
    end
end

function state(cartp::CartPole)
    return Float32[cartp.x, cartp.v, cartp.theta, cartp.theta_dot]
end

function Q̂(agent::SemiGradientSARSA, X::Vector{Float32}, A::Float32)::Float32
    return agent.Q̂(vcat(X, A))[1]
end

function Q̂_star(agent::SemiGradientSARSA, X::Vector{Float32})::Float32
    A = greedy_action(agent, X)
    return agent.Q̂(vcat(X, A))[1]
end

function learn!(agent::SemiGradientSARSA, cartpole::CartPole,
        n_episodes::Int, halfing::Int;
        t_max, termination, success, reset_cp, debug_print, snapshot=0)

    opt = Descent()
    bell_loss = 0f0
    snapshots = [(0, deepcopy(agent.Q̂),0.)]
    Δt = 1/agent.fps

    for e in 1:n_episodes
        if e % halfing == 0
            agent.ϵ /= 2
            agent.α /= 2
        end

        reset_cp(cartpole)
        X = state(cartpole)
        A = ϵ_greedy_action(agent, X) # ∈ {-1.,1.}
        done = false
        t = 0
        R = 0f0
        while !done
            force = A * cartpole.mc * 10
            r, done = step!(cartpole, force, Δt, agent.n_inter, termination=termination)
            r = Float32(r)
            R = agent.γ * R + r

            if done
                bell_loss = r - Q̂(agent, X, A)
            else
                X´ = state(cartpole)
                A´ = ϵ_greedy_action(agent, X´)
                bell_loss = r + agent.γ * Q̂(agent, X´, A´) - Q̂(agent, X, A)
            end

            ps = params(agent.Q̂)
            gs = gradient(ps) do
                Q̂(agent, X, A)
            end

            opt.eta = -agent.α * bell_loss # minus because it is substracted

            Flux.Optimise.update!(opt, ps, gs)

            if !done
                X = X´
                A = A´
            end

            t += 1
            t > t_max && break
        end

        if snapshot !=0 && (e % snapshot == 0 || success(t,R))
            push!(snapshots, (e, deepcopy(agent.Q̂), t/agent.fps))
        end

        debug_print(e, t, R)
        success(t,R) && break # considered to be solved or available time exceeded
    end

    if snapshot != 0
        return snapshots
    end
end



function learn_balance!(agent::SemiGradientSARSA, cartpole::CartPole,
        n_episodes::Int, halfing::Int=n_episodes+1; snapshot=0)

    t_min = 10_000
    t_max = 100_000
    termination = :balance
    reset_cp = reset_balance!
    debug_print(e, t, R) = println(@sprintf "Episode %d was %d steps (%.2f seconds.)" e t (t/agent.fps) )
    success(t, R) = t > t_min

    learn!(agent, cartpole, n_episodes, halfing,
            t_max=t_max, termination=:balance,
            reset_cp=reset_cp, success=success,
            debug_print=debug_print, snapshot=snapshot)
end

function learn_swingup!(agent::SemiGradientSARSA, cartpole::CartPole,
        n_episodes::Int, halfing::Int=n_episodes+1; snapshot=0)

    t_max = 10_000
    termination = :bounds
    reset_cp = reset_swingup!
end

function force(agent::SemiGradientSARSA)
    function F(cartp::CartPole)
        A = greedy_action(agent, state(cartp))
        return A * cartpole.mc * 10
    end
end
