module DifferentialGames

using LinearAlgebra


# Includes
include("test_file.jl")
include("problems/base.jl")

export 
    # Testing 
    test_f

    # Abstract Types 
    DifferentialGame 
    InverseDifferentialGame
end
