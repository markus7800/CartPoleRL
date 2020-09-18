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
end

function reset_balance!(cartp::CartPole)
    cartp.x = 0
    cartp.v = 0
    cartp.theta = π/2
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

function learn_balance!(agent::SemiGradientSARSA, cartpole::CartPole,
        n_episodes::Int, halfing::Int=n_episodes; snapshot=0)
    opt = Descent()
    bell_loss = 0f0
    snapshots = [(0, deepcopy(agent.Q̂),0.)]
    t = 0
    t_min = 10_000

    for e in 1:n_episodes
        if e % halfing == 0
            agent.ϵ /= 2
            agent.α /= 2
        end

        reset_balance!(cartpole)
        X = state(cartpole)
        A = ϵ_greedy_action(agent, X) # ∈ {-1.,1.}
        done = false
        t = 0
        while !done
            force = A * cartpole.mc * 10
            r, done = step!(cartpole, force, 1/30, agent.n_inter, termination=:balance)
            r = Float32(r)
            # custom termination
            if done
                bell_loss = r - Q̂(agent, X, A)
            else
                X´ = state(cartpole)
                A´ = ϵ_greedy_action(agent, X´)
                bell_loss = r + agent.γ * Q̂(agent, X´, A´) - Q̂(agent, X, A)
            end

            ps = params(agent.Q̂)
            gs = gradient(ps) do
                # agent.Q̂(vcat(X, A))
                Q̂(agent, X, A)
            end

            opt.eta = -agent.α * bell_loss # minus because it is substracted

            Flux.Optimise.update!(opt, ps, gs)

            if !done
                X = X´
                A = A´
            end

            t += 1
            t > 100_000 && break
        end

        if snapshot !=0 && (e % snapshot == 0 || t > t_min)
            push!(snapshots, (e, deepcopy(agent.Q̂), t/30))
        end

        println(@sprintf "Episode %d was %d steps (%.2f seconds.)" e t (t/30) )
        t > t_min && break # considered to be solved
    end

    if snapshot != 0
        return snapshots
    end
end

function force(agent::SemiGradientSARSA)
    function F(cartp::CartPole)
        A = greedy_action(agent, state(cartp))
        return A * cartpole.mc * 10
    end
end
