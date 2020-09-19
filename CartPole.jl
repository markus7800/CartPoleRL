
using Plots
using ProgressMeter
using Printf

mutable struct CartPole
    x::Float64 # cart x position
    xlims::Tuple{Float64, Float64}
    v::Float64 # cart velocity
    mc::Float64 # cart mass

    r::Float64 # pole length
    mp::Float64 # pole mass
    theta::Float64 # angle w.r.t. x-axis ∈ (-π, π]
    theta_dot::Float64 # angular velocity

end


function plot_cartpole(cartp::CartPole)
    x0, x1 = cartp.xlims
    rail_l = (x0, 0.)
    rail_r  =(x1, 0.)

    # plot rail
    p = plot([rail_l, rail_r],
            xlims=(x0-1,x1+1), ylims=(-cartp.r-.5,cartp.r+.5),
            lc=:black, legend=false, aspect_ratio=:equal)

    cart_point = (cartp.x, 0)
    pole_point = (cartp.x + cartp.r * cos(cartp.theta), cartp.r * sin(cartp.theta))

    # plot pole
    plot!([cart_point, pole_point], lc=1)

    # plot points
    scatter!([rail_l, rail_r, cart_point, pole_point],
            mc=[:black, :black, 2, 1])

    return p
end

function ∇CartPole(cartp::CartPole, f::Float64)
    mc = cartp.mc
    mp = cartp.mp
    r = cartp.r
    g = 9.81
    x = cartp.x
    x´= cartp.v
    θ = cartp.theta
    θ´ = cartp.theta_dot

    x´´ = (f - 0.5*g*mp*sin(2θ) + 0.5*mp*θ´^2*r*cos(θ)) / (mp*cos(θ)^2 + mc)
    θ´´ = (-2*g*(mp+mc)*cos(θ) + (2*f + mp*θ´^2*r*cos(θ)) * sin(θ)) / (r*mp*cos(θ) + r*mc)

    return x´´, θ´´
end

function step!(cartp::CartPole, f::Float64, Δt::Float64=1/30, n_inter::Int=1; method)
    Δ = Δt/n_inter
    x_min, x_max = cartp.xlims
    force::Float64 = f
    bounds = false
    for n in 1:n_inter
        x´´, θ´´ = ∇CartPole(cartp, force)

        # semi implicit euler
        cartp.v += Δ * x´´
        cartp.x += Δ * cartp.v

        cartp.theta_dot += Δ * θ´´
        cartp.theta += Δ * cartp.theta_dot

        if cartp.theta > π
            cartp.theta -= 2π
            @assert cartp.theta ≥ -π && cartp.theta ≤ π
        end

        # boundaries
        if cartp.x > x_max || cartp.x < x_min
            cartp.x = clamp(cartp.x, x_min, x_max)
            cartp.v = 0
            force = 0
            bounds = true
        end
    end

    fail = false
    if method == :balance
        angle = cartpole.theta < π/4 || cartpole.theta > 3/4*π
        fail = bounds || angle
    else
        fail = bounds
    end

    if fail
        r = -1.
    else
        if method == :balance
            # if pendulum is up reward = 1
            r = 0 < cartp.theta && cartp.theta < π ? 1.0 : 0.0
        elseif method == :swingup
            r = (sin(cartp.theta) + 1) / 2
        else
            r = 0
        end
    end
    return r, fail
end

function simulate_animate(cartp::CartPole, t1::Int, force::Function, n_inter::Int;
        quit_if_done=false, method=nothing, ylab="", fps=30)
    anim = Animation()
    Δt = 1/fps

    p = plot_cartpole(cartp)
    xlabel!(@sprintf "%.2f s" 0.)
    ylabel!(ylab)
    frame(anim, p)
    done = false

    @showprogress for i in 0:t1*fps
        t = i * Δt
        f = !done ? force(cartp, i, t) : 0.
        r, done = step!(cartp, f, Δt, n_inter, method=method)
        p = plot_cartpole(cartp)
        xlabel!(@sprintf "%.2f s" t)
        ylabel!(ylab)
        # annotate!(:topleft, text((@sprintf "%.2f s" t), :left))
        frame(anim, p)
        if done && quit_if_done
            break
        end
    end

    return anim
end

function simulate(cartp::CartPole, t1::Int, force::Function, n_inter::Int;
        method=nothing, fps=30)
    Δt = 1/fps

    done = false

    R = 0
    T = 0
    for i in 0:t1*fps
        t = i * Δt
        f = !done ? force(cartp, i, t) : 0.
        r, done = step!(cartp, f, Δt, n_inter, method=method)
        R += r

        if done
            T = t
            break
        end
    end

    return R, T
end

# cartpole = CartPole(0., (-2.,2.), 0., 10., 1., 1., π/2, 0.)
# plot_cartpole(cartpole)
# anim = simulate(cartpole, 10., 0., 100)
# gif(anim, "temp.gif", fps=30)

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
