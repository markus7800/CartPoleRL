include("CartPole.jl")

function time_force(F::Vector{Float64}, aps::Int, fps::Int)
    @assert fps % aps == 0
    na = fps ÷ aps
    return function force_f(cartp::CartPole, i, t)
        a = Int(i ÷ na + 1)
        if a > length(F)
            # @warn "action index to big a: $a, i: $i, t: $t"
            return 0.
        end
        # println("t: ", (@sprintf "%.4f" t), ", i: $i, a: $a, F: $(F[a])", (@sprintf ", x: %.4f, θ: %.4f" cartp.x cartp.theta))
        return F[a] * cartp.mc * 10
    end
end


# aps ... actions per second
# fps ... frames per second

function brute_swingupBFS(cartp::CartPole, aps::Int, fps::Int, t_max::Int, n_inter)
    @assert fps % aps == 0

    # F = zeros(t_max * aps)
    println("Depth = $(t_max * aps)")

    reset_swingup!(cartp)
    cartp_ = deepcopy(cartp)
    F, b = sim_backtrackBFS(cartp_, 0, t_max, fps, aps, n_inter)
    return F, b
end

function testBFS(cartp::CartPole, a::Int, A::Float64, t::Int, t_max::Int, fps::Int, aps::Int, n_inter::Int)
    Δa = 1/aps
    Δt = 1/fps
    na = fps÷aps # repeat action na times
    f = A * cartp.mc * 10

    # apply action na times
    for n in 1:na
        r, done = step!(cartp, f, Δt, n_inter, method=nothing)
        if done
            DEBUG_BACKTRACK && println(tab * "\t fail.")
            return -1, cartp
        end
    end

    t_ = t + na

    d = max(abs(cartp.x), abs(cartp.v), abs(cartp.theta - π/2), abs(cartp.theta_dot))
    if d ≤ 0.1
        return 1, cartp # goal
    elseif t_ * Δt ≥ t_max
        return -1, cartp # fail
    else
        return 0, cartp # expand
    end
end

function sim_backtrackBFS(cartp::CartPole, t::Int, t_max::Int, fps::Int, aps::Int, n_inter::Int)

    na = fps÷aps # repeat action na times

    cartp_ = deepcopy(cartp)
    nextstates = [(1, 1., deepcopy(cartp), [1.]), (1, -1., deepcopy(cartp), [-1.])]

    depth = 0

    while length(nextstates) > 0
        a, A, cartp, F = popfirst!(nextstates)
        F[a] = A
        t = na * (a-1)
        B, cartp = testBFS(cartp, a, A, t, t_max, fps, aps, n_inter)

        if a > depth
            depth = a
            println("Depth: $depth")
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
