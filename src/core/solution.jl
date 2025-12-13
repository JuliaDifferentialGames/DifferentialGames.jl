"""
    GNEPSolution{T}

General solution structure for all GNEP types (LQ games, potential games, etc.).

# Fields
- `strategies::Any` : Strategy representation (problem-dependent)
- `costs::Vector{T}` : Final costs for each player
- `metadata::Dict{Symbol, Any}` : Additional solver-specific information
"""
struct GNEPSolution{T}
    strategies::Any              # Flexible strategy storage
    costs::Vector{T}             # Cost for each player
    metadata::Dict{Symbol, Any}  # Solver-specific data
end

# Convenience constructor
function GNEPSolution(strategies, costs::Vector{T}; metadata...) where T
    GNEPSolution{T}(strategies, costs, Dict{Symbol, Any}(metadata...))
end