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

function make_objective(cartp::CartPoles, aps::Int, fps::Int, n_inter::Int, t1::Int)
    println("F should be of length ", t1 * aps)
    function objective(F::Vector{Float64})
        reset_swingup!(cartp)
        R, = simulate(cartp, t1, F, aps, n_inter, method=:swingup, fps=fps, quit_if_no_force=true)
        return -R
    end
end

using Flux

dpc = DoubleCartPole(xlims=(-2.,2.), mc=10., r_1=1., mp_1=1., theta_1=π/4, r_2=1., mp_2=1., theta_2=π/2)



obj(F)

ps = params(F)
obj(ps)

gs = gradient(obj, F)

ps[1]

using Optim
using Random
APS = 10
FPS = 30
N_INTER = 100
obj = make_objective(dpc, APS, FPS, N_INTER, 3)
Random.seed!(1)
F0 = rand(30)
obj(F0)
lower = fill(-1., 30)
upper = fill(1., 30)
@time res = optimize(obj, lower, upper, F0, SAMIN(), Optim.Options(iterations=10^6))

res.minimizer

reset_swingup!(dpc)
anim = simulate_animate(dpc, 3, time_force(res.minimizer, APS, FPS), N_INTER)
gif(anim, "temp.gif")


function GD(α=0.01, n_max = 250)
    APS = 10
    FPS = 30
    N_INTER = 100
    obj = make_objective(dpc, APS, FPS, N_INTER, 3)
    Random.seed!(1)
    F = rand(30)
    d = Inf
    score = Inf
    best = nothing
    n = 0
    while d > 0.1 && n < n_max
        n += 1
        Fx = obj(F)
        gs = gradient(obj, F)[1]
        F -= α * gs
        d = maximum(abs.(gs))
        println("$n current: ", Fx, ", gs: ", d)
        if Fx < score
            score = Fx
            best = deepcopy(F)
        end
    end
    return best
end

F = GD(0.01, 1000)


reset_swingup!(dpc)
anim = simulate_animate(dpc, 3, time_force(F, APS, FPS), N_INTER)
gif(anim, "temp.gif")
