mutable struct DoubleCartPole
    x::Float64 # cart x position
    xlims::Tuple{Float64, Float64}
    v::Float64 # cart velocity
    mc::Float64 # cart mass

    r_1::Float64 # 1st pole length
    mp_1::Float64 # 1st pole mass
    theta_1::Float64 # 1st pole angle w.r.t. x-axis ∈ (-π, π]
    theta_dot_1::Float64 # 1st pole angular velocity

    r_2::Float64 # 2nd pole length
    mp_2::Float64 # 2n pole mass
    theta_2::Float64 # 2n angle w.r.t. 1st pole ∈ (-π, π]
    theta_dot_2::Float64 # 2nd pole angular velocity

    function DoubleCartPole(;xlims, mc, r_1, r_2, mp_1, mp_2, x=0., theta_1=0., theta_2=0.)
        return new(x, xlims, 0., mc, r_1, mp_1, theta_1, 0., r_2, mp_2, theta_2, 0.)
    end
end

function plot_cartpole(cartp::DoubleCartPole)
    x0, x1 = cartp.xlims
    rail_l = (x0, 0.)
    rail_r  =(x1, 0.)

    # plot rail
    r = cartp.r_1+cartp.r_2
    p = plot([rail_l, rail_r],
            xlims=(x0-r,x1+r), ylims=(-r-.5,r+.5),
            lc=:black, legend=false, aspect_ratio=:equal)

    cart_point = (cartp.x, 0)
    pole_point_1 = (cartp.x + cartp.r_1 * cos(cartp.theta_1), cartp.r_1 * sin(cartp.theta_1))
    pole_point_2 = (pole_point_1[1] + cartp.r_2 * cos(cartp.theta_1 + cartp.theta_2),
                    pole_point_1[2] + cartp.r_2 * sin(cartp.theta_1 + cartp.theta_2))

    # plot pole
    plot!([cart_point, pole_point_1, pole_point_2], lc=1)

    # plot points
    scatter!([rail_l, rail_r, cart_point, pole_point_1, pole_point_2],
            mc=[:black, :black, 2, 1, 1])

    return p
end

function ∇CartPole(cartp::DoubleCartPole, f::Float64)
    # formulas generated by dpc_lagrange.py
    mc = cartp.mc
    mp1 = cartp.mp_1
    mp2 = cartp.mp_2

    g = 9.81

    x = cartp.x
    x´ = cartp.v

    r1 = cartp.r_1
    θ1 = cartp.theta_1
    θ1´ = cartp.theta_dot_1

    r2 = cartp.r_2
    θ2 = cartp.theta_2
    θ2´ = cartp.theta_dot_2

    x´´ = (2*f*mp1 - 4*f*mp2*cos(2*θ2) + 4*f*mp2 - g*mp1^2*sin(2*θ1) - 2*g*mp1*mp2*sin(2*θ1) + g*mp1*mp2*sin(2*θ1 + 2*θ2) +
        mp1^2*θ1´^2*r1*cos(θ1) + mp1^2*θ1´^2*r1*cos(θ1) + mp1^2*θ1´^2*r1*cos(θ1)+ 3*mp1*mp2*θ1´^2*r1*cos(θ1) -
        mp1*mp2*θ1´^2*r1*cos(θ1 + 2*θ2) + mp1*mp2*θ1´^2*r2*cos(θ1 - θ2) + 2*mp1*mp2*θ1´*θ2´*r2*cos(θ1 - θ2) + mp1*mp2*θ2´^2*r2*cos(θ1 - θ2)) /
        (mp1^2*cos(2*θ1) + mp1^2 + 2*mp1*mp2*cos(2*θ1) - 2*mp1*mp2*cos(2*θ2) - mp1*mp2*cos(2*θ1 + 2*θ2) + 3*mp1*mp2 + 2*mp1*mc - 4*mp2*mc*cos(2*θ2) + 4*mp2*mc)

    θ1´´ = (2*mp2*(g*cos(θ1 + θ2) + θ1´^2*r1*sin(θ2))*((2*r1*cos(θ2) + r2)*(mp1 + mp2 + mc) - (mp1*r1*sin(θ1) + 2*mp2*r1*sin(θ1) +
        mp2*r2*sin(θ1 + θ2))*sin(θ1 + θ2)) + r1*(mp1*sin(θ1) + 2*mp2*sin(θ1) - 2*mp2*sin(θ1 + θ2)*cos(θ2))*(2*f + mp1*θ1´^2*r1*cos(θ1) + 2*mp2*θ1´^2*r1*cos(θ1) +
        mp2*θ1´^2*r2*cos(θ1 + θ2) + 2*mp2*θ1´*θ2´*r2*cos(θ1 + θ2) + mp2*θ2´^2*r2*cos(θ1 + θ2)) - 2*(mp1 - mp2*sin(θ1 + θ2)^2 + mp2 + mc)*(g*mp1*r1*cos(θ1) +
        2*g*mp2*r1*cos(θ1) + g*mp2*r2*cos(θ1 + θ2) - 2*mp2*θ1´*θ2´*r1*r2*sin(θ2) - mp2*θ2´^2*r1*r2*sin(θ2))) /
        (r1^2*(-mp1^2*sin(θ1)^2 + mp1^2 - mp1*mp2*sin(θ1)^2 + 2*mp1*mp2*sin(θ1)*sin(θ2)*cos(θ1 + θ2) + 3*mp1*mp2*sin(θ2)^2 + mp1*mp2 + mp1*mc + 4*mp2*mc*sin(θ2)^2))

    θ2´´ = (-2*(g*cos(θ1 + θ2) + θ1´^2*r1*sin(θ2))*((mp1 + mp2 + mc)*(mp1*r1^2 + 4*mp2*r1^2 + 4*mp2*r1*r2*cos(θ2) + mp2*r2^2) - (mp1*r1*sin(θ1) +
        2*mp2*r1*sin(θ1) + mp2*r2*sin(θ1 + θ2))^2) + 2*((2*r1*cos(θ2) + r2)*(mp1 + mp2 + mc) - (mp1*r1*sin(θ1) + 2*mp2*r1*sin(θ1) +
        mp2*r2*sin(θ1 + θ2))*sin(θ1 + θ2))*(g*mp1*r1*cos(θ1) + 2*g*mp2*r1*cos(θ1) + g*mp2*r2*cos(θ1 + θ2) - 2*mp2*θ1´*θ2´*r1*r2*sin(θ2) - mp2*θ2´^2*r1*r2*sin(θ2)) -
        ((2*r1*cos(θ2) + r2)*(mp1*r1*sin(θ1) + 2*mp2*r1*sin(θ1) + mp2*r2*sin(θ1 + θ2)) - (mp1*r1^2 + 4*mp2*r1^2 + 4*mp2*r1*r2*cos(θ2) + mp2*r2^2) *
        sin(θ1 + θ2))*(2*f + mp1*θ1´^2*r1*cos(θ1) + 2*mp2*θ1´^2*r1*cos(θ1) + mp2*θ1´^2*r2*cos(θ1 + θ2) + 2*mp2*θ1´*θ2´*r2*cos(θ1 + θ2) + mp2*θ2´^2*r2*cos(θ1 + θ2))) /
        (r1^2*r2*(-mp1^2*sin(θ1)^2 + mp1^2 - mp1*mp2*sin(θ1)^2 + 2*mp1*mp2*sin(θ1)*sin(θ2)*cos(θ1 + θ2) + 3*mp1*mp2*sin(θ2)^2 + mp1*mp2 + mp1*mc + 4*mp2*mc*sin(θ2)^2))

    return x´´, θ1´´, θ2´´
end

function step!(cartp::DoubleCartPole, f::Float64, Δt::Float64=1/30, n_inter::Int=1; method)
    Δ = Δt/n_inter
    x_min, x_max = cartp.xlims
    force::Float64 = f
    bounds = false
    for n in 1:n_inter
        x´´, θ1´´, θ2´´ = ∇CartPole(cartp, force)

        # semi implicit euler
        cartp.v += Δ * x´´
        cartp.x += Δ * cartp.v

        cartp.theta_dot_1 += Δ * θ1´´
        cartp.theta_1 += Δ * cartp.theta_dot_1

        cartp.theta_dot_2 += Δ * θ2´´
        cartp.theta_2 += Δ * cartp.theta_dot_2

        if cartp.theta_1 > π
            cartp.theta_1 -= 2π
            @assert cartp.theta_1 ≥ -π && cartp.theta_1 ≤ π
        end
        if cartp.theta_2 > π
            cartp.theta_2 -= 2π
            @assert cartp.theta_2 ≥ -π && cartp.theta_2 ≤ π
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
            r = (0 < cartp.theta_1 && cartp.theta_1 < π) &&
             (-π/2 < cartp.theta_2 && cartp.theta_2 < π/2) ? 1.0 : 0.0
        elseif method == :swingup
            r = (sin(cartp.theta) + 1) / 2
        else
            r = 0
        end
    end
    return r, fail
end

function reset_balance!(cartp::DoubleCartPole)
    cartp.x = 0
    cartp.v = 0
    cartp.theta_1 = π/2
    cartp.theta_dot_1 = 0
    cartp.theta_2 = 0
    cartp.theta_dot_2 = 0
end

function reset_swingup!(cartp::DoubleCartPole)
    cartp.x = 0
    cartp.v = 0
    cartp.theta_1 = -π/2
    cartp.theta_dot_1 = 0
    cartp.theta_2 = 0
    cartp.theta_dot_2 = 0
end


# dpc = DoubleCartPole(xlims=(-2.,2.), mc=10., r_1=1., mp_1=1., theta_1=π/4, r_2=1., mp_2=1., theta_2=π/2)
# plot_cartpole(dpc)
#
# reset_balance!(dpc)
# plot_cartpole(dpc)
#
#
# reset_swingup!(dpc)
# plot_cartpole(dpc)
#
# no_force(cp, i, t) = 0.
#
# simulate(dpc, 3, no_force, 100)
#
# anim = simulate_animate(dpc, 5, no_force, 100)
# gif(anim, "doublecartpole.gif")
