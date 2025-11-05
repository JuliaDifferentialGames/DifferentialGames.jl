"""
    Player

Encapsulates one player's problem in a differential game.
"""
mutable struct Player{Y, J, f, C}
    id::Int
    x0::Vector{Float64}            # Initial state
    n::Int                         # State dimension
    m::Int                         # Control dimension
    dynamics::Union{Nothing, f}    # f(x, u, p, t)
    objective::Union{Nothing, J}   # J_i(x, u, p, t)
    constraints::Vector{C}         # C_i(x, u, p, t)
    params::Dict{Symbol, Any}      # optional player parameters
end

# Constructor
function Player(x0::AbstractVector, n::Int, m::Int; id=1, params=Dict{Symbol,Any}())
    Player{Function, Function, Function, Function}(id, x0, n, m, nothing, nothing, Function[], params)
end

function player_dynamics!(player::Player, f::Function)
    player.dynamics = f
    return player
end

function player_objective!(player::Player, J::Function)
    player.objective = J
    return player
end

function player_constraint!(player::Player, C::Function)
    push!(player.constraints, C)
    return player
end