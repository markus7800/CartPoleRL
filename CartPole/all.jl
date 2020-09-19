include("CartPole.jl")
include("DoubleCartPole.jl")

CartPoles = Union{CartPole, DoubleCartPole}

include("simulate.jl")
