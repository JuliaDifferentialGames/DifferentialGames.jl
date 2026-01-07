# ============================================================================
# CONSTRAINT SPECIFICATION
# ============================================================================

# """
#     ConstraintType

# Enumeration of constraint types for optimization problems.
# """
# @enum ConstraintType begin
#     EQUALITY        # C(x, u) = 0
#     INEQUALITY      # C(x, u) ≤ 0
#     SOC            # Second-order cone: ||Ax + b|| ≤ cᵀx + d
# end

# """
#     ConstraintSpec

# Specification for a constraint function.

# # Fields
# - `func::Function` : Constraint function
# - `type::ConstraintType` : Type of constraint
# - `dim::Int` : Output dimension of constraint function
# - `involved_players::Vector{Int}` : Which players are coupled (for shared constraints)

# # Notes
# For private constraints: `func(xⁱ, uⁱ, p, t) -> Vector{T}`
# For shared constraints: `func(X, U, p, t) -> Vector{T}` where X, U are vectors of all states/controls
# """
# struct ConstraintSpec
#     func::Function
#     type::ConstraintType
#     dim::Int
#     involved_players::Vector{Int}  # Empty for private, subset for shared
    
#     function ConstraintSpec(
#         func::Function,
#         type::ConstraintType,
#         dim::Int,
#         involved_players::Vector{Int} = Int[]
#     )
#         @assert dim > 0 "Constraint dimension must be positive"
#         new(func, type, dim, involved_players)
#     end
# end

# # Convenience constructors
# ConstraintSpec(func::Function, dim::Int; type::Symbol=:inequality) = 
#     ConstraintSpec(func, constraint_symbol_to_type(type), dim)

# function constraint_symbol_to_type(s::Symbol)
#     if s == :equality
#         return EQUALITY
#     elseif s == :inequality
#         return INEQUALITY
#     elseif s == :soc
#         return SOC
#     else
#         error("Unknown constraint type: $s. Use :equality, :inequality, or :soc")
#     end
# end

# is_private(c::ConstraintSpec) = isempty(c.involved_players)
# is_shared(c::ConstraintSpec) = !is_private(c)


"""
    ConstraintSpec

Specification for a constraint function.

# Fields
- `func::Function` : Constraint function
- `type::Symbol` : Type of constraint (:equality or :inequality)
- `dim::Int` : Output dimension of constraint function
- `involved_players::Vector{Int}` : Which players are coupled (for shared constraints)

# Notes
For private constraints: `func(xⁱ, uⁱ, p, t) -> Vector{T}`
For shared constraints: `func(X, U, p, t) -> Vector{T}` where X, U are vectors of all states/controls

# Constraint Types
- `:equality` : C(x, u) = 0
- `:inequality` : C(x, u) ≤ 0
"""
struct ConstraintSpec
    func::Function
    type::Symbol
    dim::Int
    involved_players::Vector{Int}  # Empty for private, subset for shared
    
    function ConstraintSpec(
        func::Function,
        type::Symbol,
        dim::Int,
        involved_players::Vector{Int} = Int[]
    )
        @assert dim > 0 "Constraint dimension must be positive"
        @assert type in (:equality, :inequality) "Constraint type must be :equality or :inequality"
        new(func, type, dim, involved_players)
    end
end

# Convenience constructor with default inequality type
ConstraintSpec(func::Function, dim::Int; type::Symbol=:inequality) = 
    ConstraintSpec(func, type, dim, Int[])

# Type checking functions
is_equality(c::ConstraintSpec) = (c.type == :equality)
is_inequality(c::ConstraintSpec) = (c.type == :inequality)
is_private(c::ConstraintSpec) = isempty(c.involved_players)
is_shared(c::ConstraintSpec) = !is_private(c)