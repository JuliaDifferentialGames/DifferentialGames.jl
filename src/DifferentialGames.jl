module DifferentialGames

# Includes
include("problems/base.jl")
include("problems/GNEP.jl")

# Exports
export 
    # Testing 
    test_f,

    # Abstract Types 
    DifferentialGame, 
    InverseDifferentialGame,
    AbstractLQGame,

    # Problems 
    AbstractGNEP, 
    AbstractPotentialGame, 
    AbstractLexicographicGame, 
    AbstractLQGame, 
    LQGameProblem, 
    GNEProblem, 
    PotentialGameProblem, 
    LexicographicGameProblem, 
    
    # Helper functions
    num_players, 
    state_dim, 
    control_dim

end