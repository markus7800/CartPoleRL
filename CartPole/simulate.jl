

function simulate_animate(cartp::CartPoles, t1::Int, force::Function, n_inter::Int;
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
        frame(anim, p)
        if done && quit_if_done
            break
        end
    end

    return anim
end

function simulate(cartp::CartPoles, t1::Int, force::Function, n_inter::Int;
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
