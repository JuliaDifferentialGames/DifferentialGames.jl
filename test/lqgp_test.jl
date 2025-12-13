using Test
using LinearAlgebra
using DifferentialGames  # your package module name

@testset "LQGame construction" begin
    # Define test matrices and vectors
    A = [0.0 1.0; -1.0 -0.5]
    B = [0.0; 1.0]
    Q = I(2)
    R = 1.0 * I(1)
    x0 = [1.0, 0.0]
    u0 = [0.0]

    # Construct the game
    game = LQGProblem(A, B, Q, R, x0, u0)

    # --- Type and structure checks ---
    #@test game isa LQGP
    @test game.A == A
    @test game.B == B
    @test game.Q == Q
    @test game.R == R
    @test game.x0 == x0
    @test game.u0 == u0

    # --- Dimension consistency ---
    @test size(game.A, 1) == size(game.A, 2) == length(game.x0)
    @test size(game.B, 1) == size(game.A, 1)
    @test size(game.R, 1) == size(game.B, 2)
    @test size(game.Q, 1) == size(game.A, 1)

    # # --- Sanity check on dynamic evaluation (if simulate exists) ---
    # if @isdefined simulate
    #     traj = simulate(game, (0.0, 1.0))
    #     @test all(x -> length(x) == length(x0), traj)
    # end
end