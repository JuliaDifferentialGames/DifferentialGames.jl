"""
    LQObjective(Q, R; Qf=nothing)

Quadratic LQR-style cost functional:

    J(x, u) = ∫ [ x'Qx + u'Ru ] dt + x(T)'Qf*x(T)

# Arguments
- `Q::AbstractMatrix`: State cost matrix (n×n)
- `R::AbstractMatrix`: Control cost matrix (m×m)
- `Qf::Union{Nothing, AbstractMatrix}`: Optional terminal cost matrix (n×n)

# Notes
Only Q and R appear in the integral. Qf adds a terminal penalty.
"""
struct LQObjective{T<:Real}
    Q::Matrix{T}
    R::Matrix{T}
    Qf::Union{Nothing, Matrix{T}}
end

# Constructors
LQObjective(Q::AbstractMatrix, R::AbstractMatrix) =
    LQObjective{eltype(Q)}(Q, R, nothing)

LQObjective(Q::AbstractMatrix, R::AbstractMatrix; Qf=nothing) =
    LQObjective{eltype(Q)}(Q, R, Qf)

# Instantaneous running cost (for integration)
function (J::LQObjective)(x::AbstractVector, u::AbstractVector, p, t)
    return x' * J.Q * x + u' * J.R * u
end

# Terminal cost evaluation
function terminal_cost(J::LQObjective, xT::AbstractVector)
    isnothing(J.Qf) && return zero(eltype(xT))
    return xT' * J.Qf * xT
end
