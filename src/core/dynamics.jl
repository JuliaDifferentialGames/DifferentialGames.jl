"""
    LinearDynamics(A, B; d=nothing)

Represents linear continuous-time dynamics:

    ẋ = A * x + B * u 

# Arguments
- `A::AbstractMatrix`: State transition matrix (n×n)
- `B::AbstractMatrix`: Control input matrix (n×m)
"""
struct LinearDynamics{T<:Real, F}
    A::Matrix{T}
    B::Matrix{T}
end

# Convenience constructor
LinearDynamics(A::AbstractMatrix, B::AbstractMatrix) =
    LinearDynamics{eltype(A), Nothing}(A, B, nothing)

LinearDynamics(A::AbstractMatrix, B::AbstractMatrix) =
    LinearDynamics{eltype(A)}(A, B)

# Call syntax for convenience
function (dyn::LinearDynamics)(x::AbstractVector, u::AbstractVector, p, t)
    return dyn.A * x + dyn.B * u
end