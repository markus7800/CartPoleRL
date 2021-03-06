
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

    function CartPole(;xlims, mc, r, mp, x=0., theta=0.)
        return new(x, xlims, 0., mc, r, mp, theta, 0.)
    end
end


function plot_cartpole(cartp::CartPole)
    x0, x1 = cartp.xlims
    rail_l = (x0, 0.)
    rail_r  =(x1, 0.)

    # plot rail
    p = plot([rail_l, rail_r],
            xlims=(x0-cartp.r,x1+cartp.r), ylims=(-cartp.r-.5,cartp.r+.5),
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
    # formulas generated by spc_lagrange.py
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

# cartpole = CartPole(xlims=(-1.,1.), mc=10., r=1., mp=1., theta=π/4)
# plot_cartpole(cartpole)
# no_force(cp, i, t) = 0.
#
# anim = simulate_animate(cartpole, 5, no_force, 100)
# gif(anim, "singlecartpole.gif", fps=30)
