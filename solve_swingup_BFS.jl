
function time_force(F::Vector{Float64}, aps::Int, fps::Int)
    @assert fps % aps == 0
    na = fps ÷ aps
    return function force_f(cartp, i, t)
        a = Int(i ÷ na + 1)
        if a > length(F)
            # @warn "action index to big a: $a, i: $i, t: $t"
            return 0.
        end
        # println("t: ", (@sprintf "%.4f" t), ", i: $i, a: $a, F: $(F[a])", (@sprintf ", x: %.4f, θ: %.4f" cartp.x cartp.theta))
        return F[a] * cartp.mc * 10
    end
end

mutable struct BFSSolver
    best
    score
    cartp
end

# aps ... actions per second
# fps ... frames per second

function brute_swingupBFS(cartp::CartPoles, aps::Int, fps::Int, t_max::Int, n_inter; max_depth=Inf, reset=true)
    @assert fps % aps == 0

    # F = zeros(t_max * aps)
    println("Depth = $(t_max * aps)")

    solv = BFSSolver(nothing, Inf, nothing)

    if reset
        reset_swingup!(cartp)
    end
    cartp_ = deepcopy(cartp)
    F, b = sim_backtrackBFS(solv, cartp_, 0, t_max, fps, aps, n_inter, max_depth)
    return F, b
end

function testBFS(cartp::CartPoles, a::Int, A::Float64, t::Int, t_max::Int, fps::Int, aps::Int, n_inter::Int)
    Δa = 1/aps
    Δt = 1/fps
    na = fps÷aps # repeat action na times
    f = A * cartp.mc * 10

    # apply action na times
    for n in 1:na
        r, done = step!(cartp, f, Δt, n_inter, method=nothing)
        if done
            return -1, cartp, Inf
        end
    end

    t_ = t + na

    if cartp isa CartPole
        d = max(abs(cartp.x), abs(cartp.v),
                abs(cartp.theta - π/2), abs(cartp.theta_dot))
    else
        d = max(abs(cartp.v),
                abs(cartp.theta_1 - π/2), abs(cartp.theta_dot_1),
                abs(cartp.theta_2), abs(cartp.theta_dot_2))
    end
    if d ≤ 0.1
        return 1, cartp, d # goal
    elseif t_ * Δt ≥ t_max
        return -1, cartp, d # fail
    else
        return 0, cartp, d # expand
    end
end

function sim_backtrackBFS(solv::BFSSolver, cartp::CartPoles, t::Int, t_max::Int,
    fps::Int, aps::Int, n_inter::Int, max_depth=Inf)

    na = fps÷aps # repeat action na times

    cartp_ = deepcopy(cartp)
    nextstates = [(1, 1., deepcopy(cartp), [1.]), (1, -1., deepcopy(cartp), [-1.])]

    depth = 0
    t0 = time()

    while length(nextstates) > 0
        a, A, cartp, F = popfirst!(nextstates)
        F[a] = A
        t = na * (a-1)
        B, cartp, d = testBFS(cartp, a, A, t, t_max, fps, aps, n_inter)
        if d < solv.score
            solv.best = deepcopy(F)
            solv.score = d
            solv.cartp = deepcopy(cartp)
        end

        if a > depth
            t1 = time()
            depth = a
            println("Depth: $depth, ", @sprintf "%.2fs, best: %.4f" (t1-t0) solv.score)
            t0 = t1
            if depth > max_depth
                println("Max depth exceeded...")
                println("Returning best with d=$(solv.score)")
                return solv, false
            end
        end

        if B == 1
            return F, true
        elseif B == 0
            for A´ in [1., -1.]
                F´ = deepcopy(F)
                push!(F´, A´)
                push!(nextstates, (a+1, A´, deepcopy(cartp), F´))
            end
        end
    end

    return [], false
end




# cartpole = CartPole(0, (-2.,2.), 0, 10., 1., 1., π/2, 0.)
#
# @time G, = brute_swingupBFS(cartpole, 10, 30, 3, 100)
#
# reset_swingup!(cartpole)
# anim = simulate_animate(cartpole, 3, time_force(G, 10, 30), 100)
# gif(anim, "temp.gif")
#
# plot(G)
