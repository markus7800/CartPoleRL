

""










function ODE(cp::CartPole, f::Float64)
    mc = cp.mc
    mp = cp.mp
    r = cp.r
    g = 9.81

    x_min, x_max = cp.xlims

    function F!(dX, X, p, t)
        x = X[1]
        θ = X[2]
        x´ = X[3]
        θ´ = X[4]

        dX[1] = x´
        dX[2] = θ´
        dX[3] = (f - 0.5*g*mp*sin(2θ) + 0.5*mp*θ´^2*r*cos(θ)) / (mp*cos(θ)^2 + mc)
        dX[4] = (-2*g*(mp+mc)*cos(θ) + (2*f + mp*θ´^2*r*cos(θ)) * sin(θ)) / (r*mp*cos(θ) + r*mc)
    end

    X0 = [cp.x, cp.theta, cp.v, cp.theta_dot]
    return F!, X0
end

using DifferentialEquations

function step_ODE!(cartp::CartPole, f::Float64, Δt::Float64=1/30, n_inter::Int=1)
    Δ = Δt/n_inter

    F!

    cartp.v += Δ * x´´
    cartp.x += Δ * cartp.v

    cartp.theta_dot += Δ * θ´´
    cartp.theta += Δ * cartp.theta_dot

    # if pendulum is up reward = 1
    r = 0 < cartp.theta && cartp.theta < π ? 1.0 : 0.0

    return r
end

function simulate_ODE(cartp::CartPole, t1, f)
    F!, X0 = ODE(cartp, 0.)
    tspan = (0., t1)

    F!, X0 = ODE(cartp, f)
    prob = ODEProblem(F!, X0, tspan)
    sol = solve(prob)

    anim = Animation()

    @showprogress for u in sol.u
        cartp.x = u[1]
        cartp.theta = u[2]
        cartp.v = u[3]
        cartp.theta_dot = u[4]
        p = plot_cartpole(cartp)
        frame(anim, p)
    end

    return anim
end

cartpole = CartPole(0., (-1.,1.), 0., 10., 1., 1., π/4, 0.)
anim = simulate(cartpole, 10., 0.)
gif(anim, "temp.gif", fps=15)


function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob)

plot(sol,vars=(1,2,3))




F!, X0 = ODE(cartpole, 0.)
tspan = (0., 1.0)

prob = ODEProblem(F!, X0, tspan)
sol = solve(prob)
