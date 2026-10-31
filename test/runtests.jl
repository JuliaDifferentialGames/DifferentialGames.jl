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
        @test isdefined(DifferentialGames, :SFNELQ)
        @test isdefined(DifferentialGames, :CCiLQGames)
        @test isdefined(DifferentialGames, :StochasticGameProblem)
        @test isdefined(DifferentialGames, :ChanceConstraint)
        @test isdefined(DifferentialGames, :JointRiskBudget)
        @test isdefined(DifferentialGames, :linear_observation)

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

    @testset "Stochastic games end to end (SFNELQ, shared observation, CCiLQGames)" begin
        A = [1.0 0.1; 0.0 1.0]; B = [reshape([0.0, 0.1], 2, 1)]
        g = LQGameProblem(A, B, [Matrix(1.0I(2))], [fill(0.5, 1, 1)], [Matrix(3.0I(2))], [1.0, 0.0], 5.0; dt = 1.0)
        W = [Matrix(0.02I(2)) for _ in 1:5]
        sg = StochasticGameProblem(g, ProcessNoiseModel(W); initial_covariance = Matrix(0.05I(2)))
        sol = solve(sg, SFNELQ())
        @test sol.solver_info[:status] === :solved
        @test get_expected_costs(sol)[1] > solve(g, FNELQ()).trajectories[1].cost
        obs = linear_observation([1.0 0.0], fill(0.05, 1, 1))
        sh = solve(StochasticGameProblem(g, ProcessNoiseModel(W); initial_covariance = Matrix(0.05I(2)),
                                         information = obs), SFNELQ())
        @test tr(sh.solver_info[:state_covariances][end]) > tr(sh.solver_info[:posterior_covariances][end])

        dt, N = 0.1, 10
        A2 = Matrix(1.0I(2)); B2 = [reshape([dt, 0.0], 2, 1), reshape([0.0, dt], 2, 1)]
        Q1 = [4.0 0.0; 0.0 0.0]; Q2 = [0.0 0.0; 0.0 4.0]
        g2 = LQGameProblem(A2, B2, [Q1, Q2], [fill(0.1, 1, 1), fill(0.1, 1, 1)], [Q1, Q2], [0.0, 0.0], N * dt;
                           dt = dt, q = [-(Q1 * [1.0, 0.0]), -(Q2 * [0.0, 1.0])])
        cc = ChanceConstraint(StateBounds(1; state_offset = 0, state_dim = 1, lower = [-5.0], upper = [0.6]), 0.1)
        sg2 = StochasticGameProblem(g2, ProcessNoiseModel(Matrix(0.01I(2))); chance_constraints = [cc],
                                    joint_risk = JointRiskBudget(0.1))
        cs = solve(sg2, CCiLQGames(mc_samples = 2000, seed = 1))
        @test cs.converged
        @test cs.solver_info[:joint_risk_bound] ≈ 0.1
        @test cs.solver_info[:monte_carlo].any_violation <= 0.1 + 0.03
    end
end
