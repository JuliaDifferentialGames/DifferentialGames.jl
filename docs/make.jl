# Documentation build for the DifferentialGames ecosystem.
#
# The documentation environment (docs/Project.toml) pins the three ecosystem packages through
# `[sources]` (umbrella from this checkout, Base/BaseSolvers from their `main` branches), so no
# LOAD_PATH manipulation or ad hoc `Pkg.add` is needed. For coordinated local development of the
# sibling repositories, run
#     julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path="../DifferentialGamesBase.jl"),
#                                                      PackageSpec(path="../DifferentialGamesBaseSolvers.jl")])'
# before building.

using Documenter, DocumenterCitations, DocumenterInterLinks, DocumenterLandingPage
using DifferentialGames
using DifferentialGamesBase
using DifferentialGamesBaseSolvers

DocMeta.setdocmeta!(
    DifferentialGames, :DocTestSetup,
    :(using DifferentialGames, LinearAlgebra);
    recursive=true
)

bib   = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :alpha)
links = InterLinks("Julia" => "https://docs.julialang.org/en/v1/")

makedocs(
    modules  = [DifferentialGames, DifferentialGamesBase, DifferentialGamesBaseSolvers],
    remotes  = Dict(
        pkgdir(DifferentialGamesBase) => (
            Remotes.GitHub("JuliaDifferentialGames", "DifferentialGamesBase.jl"), "main"
        ),
        pkgdir(DifferentialGamesBaseSolvers) => (
            Remotes.GitHub("JuliaDifferentialGames", "DifferentialGamesBaseSolvers.jl"), "main"
        ),
    ),
    sitename = "DifferentialGames.jl",
    build    = get(ENV, "DOCS_BUILD_DIR", "build"),   # override to keep the tracked docs/build untouched
    authors  = "BennetOutland <bennet.outland@pm.me> and contributors",
    format   = Documenter.HTML(
        prettyurls = (get(ENV, "CI", nothing) == "true") || ("--prettyurls" in ARGS) ||
                     haskey(ENV, "DOCS_BUILD_DIR"),
        canonical  = "https://JuliaDifferentialGames.github.io/DifferentialGames.jl",
        edit_link  = "main",
        assets     = ["assets/favicon.ico"],
        collapselevel = 1,
        size_threshold_warn = 200 * 2^10,
        size_threshold      = 300 * 2^10,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => [
            "Installation"        => "getting_started/installation.md",
            "A first example"     => "getting_started/first_example.md",
        ],
        "How to…" => [
            "solve an LQ game"                 => "tutorials/lq_games.md",
            "solve a nonlinear game"           => "tutorials/nonlinear_games.md",
            "handle constraints"               => "tutorials/constrained_games.md",
            "choose a solver"                  => "basics/solver_selection.md",
        ],
        "The ecosystem" => [
            "Overview"                  => "ecosystem/overview.md",
            "Cross-package conventions" => "ecosystem/conventions.md",
            "Benchmarks"                => "ecosystem/benchmarks.md",
        ],
        "Benchmark suite" => [
            "Overview"                  => "benchmarks/overview.md",
            "Hallway passing"           => "benchmarks/hallway.md",
            "Racing"                    => "benchmarks/racing.md",
            "Multi-agent particle tasks" => "benchmarks/particle_tasks.md",
        ],
        "Mathematical background" => [
            "Differential games"        => "math/differential_games.md",
            "Equilibrium concepts"      => "math/equilibrium_concepts.md",
            "Information structures"     => "math/information_structures.md",
            "Optimality conditions"     => "math/optimality_conditions.md",
            "Numerical methods"         => "math/numerical_methods.md",
        ],
        "Problem types" => [
            "Overview"                       => "problem_types/overview.md",
            "LQGameProblem"                  => "problem_types/lq_game.md",
            "LTVLQGameProblem"               => "problem_types/ltv_lq_game.md",
            "PDGNEProblem"                   => "problem_types/pdgnep.md",
            "LexicographicGameProblem"       => "problem_types/lexicographic.md",
            "ConvexGameProblem"              => "problem_types/convex.md",
            "Potential games"                => "problem_types/potential.md",
            "Stochastic, robust, partially observable" => "problem_types/uncertainty.md",
            "InverseGameProblem"             => "problem_types/inverse_game.md",
        ],
        "Solvers" => [
            "FNELQ"               => "solvers/fnelq.md",
            "SFNELQ"              => "solvers/sfnelq.md",
            "iLQGames"            => "solvers/ilqgames.md",
            "ALGAMES"             => "solvers/algames.md",
            "DGSQP"               => "solvers/dgsqp.md",
            "FALCON"              => "solvers/falcon.md",
            "InverseLQGames"      => "solvers/invlq.md",
            "LIBR"                => "solvers/libr.md",
            "RecedingHorizonNash" => "solvers/rhn.md",
            "YiPavel"             => "solvers/yipavel.md",
            "DynamicTubeMPC"      => "solvers/dtmpc.md",
            "Writing a solver page" => "basics/solver_page_template.md",
        ],
        "Multi-agent reinforcement learning" => [
            "DifferentialGamesMARL"  => "marl/overview.md",
            "Environments and tasks" => "marl/environments.md",
            "MADDPG"                 => "marl/maddpg.md",
            "Guided policy search"   => "marl/gps.md",
        ],
        "Features" => [
            "Overview & architecture" => "basics/overview.md",
            "Building problems"       => "basics/problem_interface.md",
            "Working with solutions"  => "basics/solutions.md",
            "Common solver options"   => "basics/solver_options.md",
        ],
        "API reference" => [
            "Problem types"      => "api/problems.md",
            "Dynamics"           => "api/dynamics.md",
            "Costs & objectives" => "api/costs.md",
            "Constraints"        => "api/constraints.md",
            "Solutions"          => "api/solutions.md",
            "Solver interface"   => "api/solver_interface.md",
            "Inverse games"      => "api/inverse_games.md",
        ],
        "Miscellanea" => [
            "About"        => "misc/about.md",
            "Contributing" => "contributing.md",
            "Notation"     => "misc/notation.md",
            "References"   => "misc/references.md",
        ],
    ],
    plugins = [bib, links, LandingPage()],
    # Strict: broken references, failing @example/@repl blocks and failing doctests fail the build.
    checkdocs = :none,   # TODO(docs-audit): switch to :exports once docstring coverage across Base/BaseSolvers is complete
    warnonly  = false,
    doctest   = true,
)

deploydocs(
    repo      = "github.com/JuliaDifferentialGames/DifferentialGames.jl",
    devbranch = "main",
)
