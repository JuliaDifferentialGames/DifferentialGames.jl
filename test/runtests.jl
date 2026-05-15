using DifferentialGames
using Test
using LinearAlgebra

@time @testset "DifferentialGames.jl" begin

    @testset "Re-exports" begin
        # Core problem types from DifferentialGamesBase
        @test isdefined(DifferentialGames, :GameProblem)
        @test isdefined(DifferentialGames, :LQGameProblem)
        @test isdefined(DifferentialGames, :PDGNEProblem)
        @test isdefined(DifferentialGames, :PlayerSpec)
        @test isdefined(DifferentialGames, :PlayerObjective)
        @test isdefined(DifferentialGames, :LQStageCost)
        @test isdefined(DifferentialGames, :LQTerminalCost)
        @test isdefined(DifferentialGames, :DiagonalLQStageCost)
        @test isdefined(DifferentialGames, :DiagonalLQTerminalCost)
        @test isdefined(DifferentialGames, :GNEPSolution)

        # IGNEP types
        @test isdefined(DifferentialGames, :InverseGameProblem)
        @test isdefined(DifferentialGames, :InversePDGNEProblem)
        @test isdefined(DifferentialGames, :ForwardSolverWrapper)
        @test isdefined(DifferentialGames, :KnownObjective)
        @test isdefined(DifferentialGames, :UnknownObjective)
        @test isdefined(DifferentialGames, :FullStateObservation)
        @test isdefined(DifferentialGames, :NoisyObservation)
        @test isdefined(DifferentialGames, :ObservationData)
        @test isdefined(DifferentialGames, :InverseGameSolution)

        # Solvers from DifferentialGamesBaseSolvers
        @test isdefined(DifferentialGames, :FNELQ)
        @test isdefined(DifferentialGames, :iLQGames)
        @test isdefined(DifferentialGames, :ALGAMES)

        # solve interface
        @test isdefined(DifferentialGames, :solve)
    end

    @testset "FNELQ — 2-player LQ game" begin
        n, m = 4, 2
        A = diagm([0.9, 0.9, 0.9, 0.9])
        B = [reshape([1.0, 0.0, 0.0, 0.0], n, 1),
             reshape([0.0, 0.0, 1.0, 0.0], n, 1)]
        Q  = [diagm(ones(n)), diagm(ones(n))]
        R  = [reshape([0.1], 1, 1), reshape([0.1], 1, 1)]
        Qf = Q
        x0 = ones(n)

        game = LQGameProblem(A, B, Q, R, Qf, x0, 1.0; dt=0.1)
        @test game isa GameProblem
        @test num_players(game) == 2
        @test n_steps(game) == 10

        sol = solve(game, FNELQ())
        @test sol isa GNEPSolution
        @test sol.converged
        @test get_cost(sol, 1) >= 0
        @test get_cost(sol, 2) >= 0
    end

    @testset "FNELQ — single-player reduces to LQR" begin
        n, m = 2, 1
        A = [1.0 0.1; 0.0 1.0]
        B = [reshape([0.0, 0.1], n, 1)]
        Q  = [diagm([1.0, 0.1])]
        R  = [reshape([0.01], 1, 1)]
        Qf = [diagm([10.0, 1.0])]
        x0 = [1.0, 0.0]

        game = LQGameProblem(A, B, Q, R, Qf, x0, 2.0; dt=0.1)
        sol = solve(game, FNELQ())
        @test sol.converged
        @test get_cost(sol, 1) >= 0
    end

    @testset "ALGAMES — constrained 2-player game" begin
        n, m = 4, 2
        A = diagm([0.95, 0.95, 0.95, 0.95])
        B = [reshape([1.0, 0.0, 0.0, 0.0], n, 1),
             reshape([0.0, 0.0, 1.0, 0.0], n, 1)]
        Q  = [diagm(ones(n)), diagm(ones(n))]
        R  = [reshape([0.1], 1, 1), reshape([0.1], 1, 1)]
        Qf = Q
        x0 = [1.0, 0.0, -1.0, 0.0]

        game = LQGameProblem(A, B, Q, R, Qf, x0, 1.0; dt=0.1)
        sol = solve(game, ALGAMES())
        @test sol isa GNEPSolution
        @test get_cost(sol, 1) >= 0
        @test get_cost(sol, 2) >= 0
    end

    @testset "PDGNEProblem construction via DifferentialGames" begin
        n, m = 4, 2
        x0 = [1.0, 0.0, 0.0, 0.0]

        dynamics = (xi, ui, p, t) -> [xi[3]; xi[4]; ui[1]; ui[2]]
        stage    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
        terminal = DiagonalLQTerminalCost(10.0 * ones(n))
        obj      = PlayerObjective(1, stage, terminal)
        player   = PlayerSpec(1, n, m, x0, dynamics, obj)

        game = PDGNEProblem([player], 2.0, 0.1)
        @test game isa GameProblem
        @test num_players(game) == 1
        @test is_pd_gnep(game)
        @test state_dim(game) == n
    end

end
