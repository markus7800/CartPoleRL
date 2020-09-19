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
        println("t: ", (@sprintf "%.4f" t), ", i: $i, a: $a, F: $(F[a])", (@sprintf ", x: %.4f, θ: %.4f" cartp.x cartp.theta))
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
function brute_swingup(cartp::CartPole, aps::Int, fps::Int, t_max::Int, n_inter)
    @assert fps % aps == 0

    F = zeros(t_max * aps)
    println("Depth = $(length(F))")

    reset_swingup!(cartp)
    cartp_ = deepcopy(cartp)
    for A in [-1, 1]
        F[1] = A
        b = sim_backtrack(cartp_, F, 0, t_max, fps, aps, n_inter)
        if b
            return F, true
        end
    end
    return F, false
end

global best = Inf
global DEBUG_BACKTRACK = false
function sim_backtrack(cartp::CartPole, F::Vector{Float64}, t::Int, t_max::Int, fps::Int, aps::Int, n_inter::Int)
    Δa = 1/aps
    Δt = 1/fps
    na = fps÷aps # repeat action na times

    a = t ÷ na + 1
    f = F[a] * cartp.mc * 10
    @assert f != 0


    tab = "\t"^(a-1)
    DEBUG_BACKTRACK && println(tab * "a: $a f: $f ts: ", (@sprintf "%.4f:%.4f; x: %.4f, θ: %.4f" t*Δt (t*Δt + Δt*(na-1)) cartp.x cartp.theta))

    # apply action aps times
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
    if d ≤ best
        global best = d
        println("best with $d", F)
    end
    if d ≤ 0.1
        return true
    elseif t_ * Δt ≥ t_max
        # println(tab * "\tθ: ", cartp.theta)
        return false
    else
        cartp_ = deepcopy(cartp)
        for A in [-1, 1]
            F[a+1] = A
            cartp.x = cartp_.x; cartp.v = cartp_.v;
            cartp.theta = cartp_.theta; cartp.theta_dot = cartp_.theta_dot
            b = sim_backtrack(cartp, F, t_, t_max, fps, aps, n_inter)
            if b
                return true
            end
        end
    end
    return false
end



cartpole = CartPole(0, (-2.,2.), 0, 10., .5, 1., π/2, 0.)



brute_swingup(cartpole, 2, 4, 3, 100)
brute_swingup(cartpole, 2, 30, 3, 100)

@time F, = brute_swingup(cartpole, 10, 30, 3, 100)

reset_swingup!(cartpole)
simulate(cartpole, 3, force(F, 10, 30), 100)

reset_swingup!(cartpole)
anim = simulate_animate(cartpole, 3, force(F, 10, 30), 100)

gif(anim, "temp.gif")


brute_swingup(cartpole, 2, 30, 3, 100)

F = [-1., 1., -1., -1., -1., 1., -1.]
reset_swingup!(cartpole)
anim = simulate(cartpole, 3, force(F, 2, 30), 100)

# a: 1 f: -100.0 ts: 0.0000:0.4667; x: 0.0000, θ: -1.5708
#     a: 2 f: 100.0 ts: 0.5000:0.9667; x: -2.0000, θ: -0.8803
#         a: 3 f: -100.0 ts: 1.0000:1.4667; x: -0.8746, θ: -3.4079
#             a: 4 f: -100.0 ts: 1.5000:1.9667; x: 0.2051, θ: 2.6128
#                 a: 5 f: -100.0 ts: 2.0000:2.4667; x: -0.9797, θ: 2.8203
#                      fail.
#                 a: 5 f: 100.0 ts: 2.0000:2.4667; x: -1.9986, θ: -0.3926
#                     a: 6 f: -100.0 ts: 2.5000:2.9667; x: -0.6764, θ: 2.5664
#                     a: 6 f: 100.0 ts: 2.5000:2.9667; x: 0.5947, θ: 1.8137
