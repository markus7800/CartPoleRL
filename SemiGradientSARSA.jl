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

function state(cartp::DoubleCartPole)
    return Float32[cartp.x, cartp.v, cartp.theta_1, cartp.theta_dot_1, cartp.theta_2, cartp.theta_dot_2]
end

function Q̂(agent::SemiGradientSARSA, X::Vector{Float32}, A::Float32)::Float32
    return agent.Q̂(vcat(X, A))[1]
end

function Q̂_star(agent::SemiGradientSARSA, X::Vector{Float32})::Float32
    A = greedy_action(agent, X)
    return agent.Q̂(vcat(X, A))[1]
end

function learn!(agent::SemiGradientSARSA, cartpole::CartPoles,
        n_episodes::Int, halfing::Int;
        t_max, method, success, reset_cp, debug_print, snapshot=0)
    opt = Descent()
    bell_loss = 0f0
    snapshots = [(0, deepcopy(agent.Q̂),0., 0.)]
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
            t += 1

            force = A * cartpole.mc * 10
            r, done = step!(cartpole, force, Δt, agent.n_inter, method=method)
            r = Float32(r)
            R = R + r

            if done
                bell_loss = r - Q̂(agent, X, A)
                # println("t: $t X: $X A: $A R: ", r)
            else
                X´ = state(cartpole)
                A´ = ϵ_greedy_action(agent, X´)
                bell_loss = r + agent.γ * Q̂(agent, X´, A´) - Q̂(agent, X, A)
                # println("t: $t X: $X A: $A R: ", r + agent.γ * Q̂(agent, X´, A´))
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

            t > t_max && break
        end

        if snapshot !=0 && (e % snapshot == 0 || success(t,R))
            push!(snapshots, (e, deepcopy(agent.Q̂), t/agent.fps, R))
        end

        debug_print(e, t, R)
        success(t,R) && break # considered to be solved or available time exceeded
    end

    if snapshot != 0
        return snapshots
    end
end



function learn_balance!(agent::SemiGradientSARSA, cartpole::CartPoles,
        n_episodes::Int, halfing::Int=n_episodes+1; snapshot=0)

    goal_time = 10_000
    t_max = 100_000
    method = :balance
    reset_cp = reset_balance!
    debug_print(e, t, R) = println(@sprintf "Episode %d was %d steps (%.2f seconds.)" e t (t/agent.fps) )
    success(t, R) = t > goal_time

    learn!(agent, cartpole, n_episodes, halfing,
            t_max=t_max, method=method,
            reset_cp=reset_cp, success=success,
            debug_print=debug_print, snapshot=snapshot)
end

function learn_swingup!(agent::SemiGradientSARSA, cartpole::CartPoles,
        n_episodes::Int, halfing::Int=n_episodes+1; snapshot=0)

    t_max = 10_000
    method = :swingup
    reset_cp = reset_swingup!
    debug_print(e, t, R) = println(@sprintf "Episode %d yields %.2f reward " e R)

    max_reward = 1/(1-agent.γ) * t_max # but impossible

    swing_up_time = 1_000
    goal_reward = (t_max-swing_up_time)

    success(t, R) = R > goal_reward

    learn!(agent, cartpole, n_episodes, halfing,
            t_max=t_max, method=method,
            reset_cp=reset_cp, success=success,
            debug_print=debug_print, snapshot=snapshot)
end

function force(agent::SemiGradientSARSA)
    function F(cartp::CartPoles, i, t)
        A = greedy_action(agent, state(cartp))
        return A * cartp.mc * 10
    end
end






#================= N STEP ===================#

function learn_n_step!(agent::SemiGradientSARSA, cartpole::CartPoles,
        N::Int, n_episodes::Int, halfing::Int;
        t_max, method, success, reset_cp, debug_print, snapshot=0)
    opt = Descent()
    bell_loss = 0f0
    snapshots = [(0, deepcopy(agent.Q̂),0., 0.)]
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
        t = 1
        T = t_max
        R = 0f0

        Rs = [0f0]; Xs = [X]; As = [A]
        #=
        reward is stored offset
        X_0 A_0 _
        X_1 A_1 R_1
        ...
        X_T A_T R_T
        where R_1 is reward after taking action A_0
        updates up to X_{T-1} A_{T-1}
        =#
        while true
            # println("\nt: $t T: $T")
            if t < T
                force = A * cartpole.mc * 10
                r, done = step!(cartpole, force, Δt, agent.n_inter, method=method)
                r = Float32(r)
                R = R + r
                push!(Rs, r)

                if done
                    T = t+1
                else
                    X = state(cartpole)
                    A = ϵ_greedy_action(agent, X)
                    push!(Xs, X), push!(As, A)
                end
            end

            τ = t - N + 1

            if τ ≥ 1
                # println("t: $t τ: $τ T: $T")
                G = sum((agent.γ^(i-τ-1)) * Rs[i] for i in τ+1:min(τ+N, T))
                rewards_used = collect(τ+1:min(τ+N,T))
                # println("rewards used: ", rewards_used)
                if τ + N < T
                    G += agent.γ^N * Q̂(agent, Xs[τ+N], As[τ+N])
                    # println("bootstrapped: ", τ+N)
                end
                # print("X: $(Xs[τ]), A: $(As[τ]) G: $G\n")

                ps = params(agent.Q̂)
                gs = gradient(ps) do
                    Q̂(agent, Xs[τ], As[τ])
                end
                bell_loss = (G - Q̂(agent, Xs[τ], As[τ]))
                opt.eta = -agent.α * bell_loss # minus because it is substracted

                Flux.Optimise.update!(opt, ps, gs)
            end
            if τ == T-1
                # println("natural break")
                break
            end
            t += 1
        end

        if snapshot !=0 && (e % snapshot == 0 || success(t,R))
            push!(snapshots, (e, deepcopy(agent.Q̂), t/agent.fps, R))
        end

        debug_print(e, T-1, R)
        success(T-1,R) && break # considered to be solved or available time exceeded
    end

    if snapshot != 0
        return snapshots
    end
end

function learn_balance_n_step!(agent::SemiGradientSARSA, cartpole::CartPoles,
        N::Int, n_episodes::Int, halfing::Int=n_episodes+1; snapshot=0)

    goal_time = 10_000
    t_max = 100_000
    method = :balance
    reset_cp = reset_balance!
    debug_print(e, t, R) = println(@sprintf "Episode %d was %d steps (%.2f seconds.)" e t (t/agent.fps) )
    success(t, R) = t > goal_time

    learn_n_step!(agent, cartpole, N, n_episodes, halfing,
            t_max=t_max, method=method,
            reset_cp=reset_cp, success=success,
            debug_print=debug_print, snapshot=snapshot)
end
