include("CartPole.jl")

function force(F::Vector{Float64}, aps::Int, fps::Int)
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

begin
    Δa = 1/5
    Δt = 1/30

    for (i,t) in enumerate(0:Δt:3)
        a = t ÷ Δa + 1
        println("i: $i, t: $t a: $a")
    end
end


# aps ... actions per second
# fps ... frames per second
function brute_swingupDFS(cartp::CartPole, aps::Int, fps::Int, t_max::Int, n_inter)
    @assert fps % aps == 0

    F = zeros(t_max * aps)
    println("Depth = $(length(F))")

    reset_swingup!(cartp)
    cartp_ = deepcopy(cartp)
    for A in [-1, 1]
        F[1] = A
        cartp.x = cartp_.x; cartp.v = cartp_.v;
        cartp.theta = cartp_.theta; cartp.theta_dot = cartp_.theta_dot
        b = sim_backtrackDFS(cartp, F, 0, t_max, fps, aps, n_inter)
        if b
            return F, true
        end
    end
    return F, false
end

function sim_backtrackDFS(cartp::CartPole, F::Vector{Float64}, t::Int, t_max::Int, fps::Int, aps::Int, n_inter::Int)
    Δa = 1/aps
    Δt = 1/fps
    na = fps÷aps # repeat action na times

    a = t ÷ na + 1
    f = F[a] * cartp.mc * 10
    @assert f != 0


    tab = "\t"^(a-1)
    DEBUG_BACKTRACK && println(tab * "a: $a f: $f ts: ", (@sprintf "%.4f:%.4f; x: %.4f, θ: %.4f" t*Δt (t*Δt + Δt*(na-1)) cartp.x cartp.theta))

    # apply action na times
    for n in 1:na
        r, done = step!(cartp, f, Δt, n_inter, method=nothing)
        if done
            DEBUG_BACKTRACK && println(tab * "\t fail.")
            return false
        end
    end
    t_ = t + na
    a2 = Int(t_ ÷ na + 1)
    @assert a + 1 == a2 "$a $a2 $t $t_" # action for next time step should be next action

    d = max(abs(cartp.v), abs(cartp.theta - π/2), abs(cartp.theta_dot))
    if d ≤ 0.1
        return true
    elseif t_ * Δt ≥ t_max
        return false
    else
        cartp_ = deepcopy(cartp)
        for A in [-1, 1]
            F[a+1] = A
            cartp.x = cartp_.x; cartp.v = cartp_.v;
            cartp.theta = cartp_.theta; cartp.theta_dot = cartp_.theta_dot
            b = sim_backtrackDFS(cartp, F, t_, t_max, fps, aps, n_inter)
            if b
                return true
            end
        end
    end
    return false
end

function brute_swingupBFS(cartp::CartPole, aps::Int, fps::Int, t_max::Int, n_inter)
    @assert fps % aps == 0

    # F = zeros(t_max * aps)
    println("Depth = $(t_max * aps)")

    reset_swingup!(cartp)
    cartp_ = deepcopy(cartp)
    F, b = sim_backtrackBFS(cartp_, 0, t_max, fps, aps, n_inter)
    return F, b
end

function testBFS(cartp::CartPole, F::Vector{Float64}, t::Int, t_max::Int, fps::Int, aps::Int, n_inter::Int)
    Δa = 1/aps
    Δt = 1/fps
    na = fps÷aps # repeat action na times

    a = t ÷ na + 1
    f = F[a] * cartp.mc * 10
    @assert f != 0


    tab = "\t"^(a-1)
    DEBUG_BACKTRACK && println(tab * "a: $a F: $(F[1:a]) ts: ", (@sprintf "%.4f:%.4f; x: %.4f, θ: %.4f" t*Δt (t*Δt + Δt*(na-1)) cartp.x cartp.theta))

    # apply action na times
    for n in 1:na
        r, done = step!(cartp, f, Δt, n_inter, method=nothing)
        if done
            DEBUG_BACKTRACK && println(tab * "\t fail.")
            return -1, cartp
        end
    end
    t_ = t + na
    a2 = Int(t_ ÷ na + 1)
    @assert a + 1 == a2 "$a $a2 $t $t_" # action for next time step should be next action

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
        B, cartp = testBFS(cartp, F, t, t_max, fps, aps, n_inter)

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



cartpole = CartPole(0, (-2.,2.), 0, 10., .5, 1., π/2, 0.)



global DEBUG_BACKTRACK = false

brute_swingupDFS(cartpole, 2, 4, 3, 100)

@time F, = brute_swingupDFS(cartpole, 10, 30, 2, 100)

reset_swingup!(cartpole)
simulate(cartpole, 2, force(F, 10, 30), 100)

reset_swingup!(cartpole)
anim = simulate_animate(cartpole, 2, force(F, 10, 30), 100)
gif(anim, "temp.gif")


cartpole = CartPole(0, (-2.,2.), 0, 10., 1., 1., π/2, 0.)

brute_swingupBFS(cartpole, 5, 30, 2, 100)

@time G, = brute_swingupBFS(cartpole, 10, 30, 3, 100)

reset_swingup!(cartpole)
anim = simulate_animate(cartpole, 3, force(G, 10, 30), 100)
gif(anim, "temp.gif")

plot(G)
