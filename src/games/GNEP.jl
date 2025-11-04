"""
AbstractGNEP{T, Y, J, f, C} <: DifferentialGame

Abstract base type for a Generalized Nash Equilibrium Problem.

# Type parameters
- `T` : time horizon type (e.g., `Float64`, `Int`, or `AbstractTimeRange`)
- `Y` : strategy type (e.g., vector of controls, functions)
- `J` : payoff/cost functional type
- `f` : dynamics type (e.g., function `f(t, x, u)`)
- `C` : constraints type (e.g., vector of functions, sets)
"""
abstract type AbstractGNEP{T, Y, J, f, C} <: DifferentialGame end


"""
AbstractLQGame{T, Y, J, f, C} <: GNEP{T, Y, J, f, C}

Abstract base type for a linear-quadratic game.

# Type parameters
- `T` : time horizon type (e.g., `Float64`, `Int`, or `AbstractTimeRange`)
- `Y` : strategy type (e.g., vector of controls, functions)
- `J` : payoff/cost functional type
- `f` : dynamics type (e.g., function `f(t, x, u)`)
- `C` : constraints type (e.g., vector of functions, sets)
"""
abstract type AbstractLQGame{T, Y, J, f, C} <: AbstractGNEP{T, Y, J, f, C} end


"""
    LQGame

Concrete implementation of a linear-quadratic differential game.

# Fields
- `A::Matrix{Float64}` : system dynamics matrix
- `B::Matrix{Float64}` : control input matrix
- `Q::Matrix{Float64}` : state cost matrix
- `R::Matrix{Float64}` : control cost matrix
- `x0::Vector{Float64}` : initial state
- `u0::Vector{Float64}` : initial control

# Notes
This type implements `AbstractLQGame` and is suitable for use with LQ solvers.
"""
struct LQGame <: AbstractLQGame{Float64, Vector{Float64}, Function, Function, Nothing}
    A::Matrix{Float64}  # n×n
    B::Matrix{Float64}  # n×m
    Q::Matrix{Float64}  # n×n
    R::Matrix{Float64}  # m×m
    x0::Vector{Float64} # n
    u0::Vector{Float64} # m
end




abstract type AbstractZeroSumGame{T,Y,J,f,C} <: AbstractGNEP{T,Y,J,f,C} end
abstract type AbstractPotentialGame{T,Y,J,f,C} <: AbstractGNEP{T,Y,J,f,C} end
abstract type AbstractLexicographicalGame{T,Y,J,f,C} <: AbstractPotentialGame{T,Y,J,f,C} end

