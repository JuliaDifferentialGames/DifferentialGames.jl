using Pkg
Pkg.activate(@__DIR__)         # activate experiments/ environment
Pkg.develop(path=joinpath(@__DIR__, ".."))  # point to your main package
Pkg.instantiate()

using Revise
using LinearAlgebra
using DifferentialGames

# Example 1: Two-player pursuit-evasion game
player1 = Player(
    id=1,
    n=4,  # [x, y, vx, vy]
    m=2,  # [ax, ay]
    x0=[0.0, 0.0, 1.0, 0.0],
    dynamics=(xⁱ, uⁱ, p, t) -> [xⁱ[3], xⁱ[4], uⁱ[1], uⁱ[2]],
    running_cost=(X, uⁱ, p, t) -> norm(X[1][1:2] - X[2][1:2])^2 + 0.1*norm(uⁱ)^2,
    terminal_cost=(X) -> 10.0*norm(X[1][1:2] - X[2][1:2])^2
)

player2 = Player(
    id=2,
    n=4,
    m=2,
    x0=[5.0, 5.0, -1.0, 0.0],
    dynamics=(xⁱ, uⁱ, p, t) -> [xⁱ[3], xⁱ[4], uⁱ[1], uⁱ[2]],
    running_cost=(X, uⁱ, p, t) -> -norm(X[1][1:2] - X[2][1:2])^2 + 0.1*norm(uⁱ)^2,
    terminal_cost=(X) -> -10.0*norm(X[1][1:2] - X[2][1:2])^2
)

# # Collision avoidance
# collision = ConstraintSpec(
#     (X, U, p, t) -> [0.5 - norm(X[1][1:2] - X[2][1:2])],
#     INEQUALITY,
#     1,
#     [1, 2]
# )

# # Define the game
# game = PDGNEProblem([player1, player2], [collision], 10.0, 0.1)