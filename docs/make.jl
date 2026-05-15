using Documenter
using DifferentialGames
using DifferentialGamesBase
using DifferentialGamesBaseSolvers

DocMeta.setdocmeta!(
    DifferentialGames, :DocTestSetup,
    :(using DifferentialGames, LinearAlgebra);
    recursive=true
)

makedocs(
    modules  = [DifferentialGames, DifferentialGamesBase, DifferentialGamesBaseSolvers],
    sitename = "DifferentialGames.jl",
    authors  = "BennetOutland <bennet.outland@pm.me> and contributors",
    format   = Documenter.HTML(
        canonical        = "https://JuliaDifferentialGames.github.io/DifferentialGames.jl",
        edit_link        = "main",
        assets           = String[],
        sidebar_sitename = true,
        collapselevel    = 1,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation"   => "getting_started/installation.md",
            "First Example"  => "getting_started/first_example.md",
        ],
        "Tutorials" => [
            "LQ Games"              => "tutorials/lq_games.md",
            "Nonlinear Games"       => "tutorials/nonlinear_games.md",
            "Constrained Games"     => "tutorials/constrained_games.md",
            "Inverse Games"         => "tutorials/inverse_games.md",
        ],
        "Basics" => [
            "Overview & Architecture" => "basics/overview.md",
            "Building Problems"       => "basics/problem_interface.md",
            "Working with Solutions"  => "basics/solutions.md",
            "Common Solver Options"   => "basics/solver_options.md",
        ],
        "Problem Types" => [
            "LQGameProblem"          => "problem_types/lq_game.md",
            "LTVLQGameProblem"       => "problem_types/ltv_lq_game.md",
            "PDGNEProblem"           => "problem_types/pdgnep.md",
            "InverseGameProblem"     => "problem_types/inverse_game.md",
        ],
        "Solver Algorithms" => [
            "FNELQ"    => "solvers/fnelq.md",
            "iLQGames" => "solvers/ilqgames.md",
            "ALGAMES"  => "solvers/algames.md",
        ],
        "API Reference" => [
            "Problem Types"      => "api/problems.md",
            "Dynamics"           => "api/dynamics.md",
            "Costs & Objectives" => "api/costs.md",
            "Constraints"        => "api/constraints.md",
            "Solutions"          => "api/solutions.md",
            "Solver Interface"   => "api/solver_interface.md",
            "Inverse Games"      => "api/inverse_games.md",
        ],
        "Contributing" => "contributing.md",
    ],
    checkdocs = :none,
    warnonly  = true,
)

deploydocs(
    repo      = "github.com/JuliaDifferentialGames/DifferentialGames.jl",
    devbranch = "main",
)
