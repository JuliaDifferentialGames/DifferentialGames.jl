# Contributing

Contributions are welcome. This page covers how to set up a development environment, run the test suite, and submit changes.

## Reporting Issues

Please file bugs and feature requests on the [GitHub issue tracker](https://github.com/JuliaDifferentialGames/DifferentialGames.jl/issues). Include:

- Julia version (`julia --version`)
- Package version (`Pkg.status("DifferentialGames")`)
- A minimal reproducible example

## Development Setup

DifferentialGames.jl is an umbrella package; most code lives in the sub-packages. Clone all three:

```bash
git clone https://github.com/JuliaDifferentialGames/DifferentialGames.jl
git clone https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl
git clone https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl
```

Then dev them into a shared environment:

```julia
using Pkg
Pkg.develop([
    PackageSpec(path="path/to/DifferentialGamesBase.jl"),
    PackageSpec(path="path/to/DifferentialGamesBaseSolvers.jl"),
    PackageSpec(path="path/to/DifferentialGames.jl"),
])
```

## Running Tests

Each package has its own test suite:

```bash
# From within the package directory:
julia --project -e 'using Pkg; Pkg.test()'

# Or directly:
julia --project test/runtests.jl
```

Expected pass counts (approximate):

| Package | Tests |
|---------|-------|
| `DifferentialGamesBase` | ~1100 |
| `DifferentialGamesBaseSolvers` | ~260 |
| `DifferentialGames` | ~40 |

## Code Style

- Follow the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/).
- Use 4-space indentation.
- Type-annotate public function signatures.
- Add docstrings for all exported names using the standard Julia format:

```julia
"""
    my_function(x, y) -> z

One-line summary.

Longer description if needed.
"""
function my_function(x, y)
    ...
end
```

## Adding a New Solver

New solvers belong in `DifferentialGamesBaseSolvers.jl`. The required steps are:

1. Create `src/solvers/MySolver/` with at least `MySolver.jl` and `test/mysolver_tests.jl`.
2. Implement `solve(prob, ::MySolver; ...)` returning a `GNEPSolution`.
3. Implement `solver_capabilities(::Type{MySolver})` returning a `Vector{Symbol}`.
4. Export the solver type and include the file in `src/DifferentialGamesBaseSolvers.jl`.
5. Include the test file in `test/runtests.jl`.
6. Add a documentation page under `DifferentialGames.jl/docs/src/solvers/`.

Note, the above code does not need to be merged in with `DifferentialGamesBaseSolvers.jl` if you wish to keep it in your own repo. You are still free to add the documenation to `DifferentialGamesBaseSolvers.jl` to improve the visibility of your repo. 

## Adding a New Problem Type

New problem types belong in `DifferentialGamesBase.jl`. A problem type must:

1. Be a concrete subtype of `AbstractGameProblem` (or define a new abstract type).
2. Have an appropriate solution type (`AbstractGameSolution` subtype).
3. Export the type and any constructors.
4. Include `@docs` blocks in the relevant `DifferentialGames.jl` API reference page.

## Pull Request Guidelines

- One logical change per PR.
- All tests must pass.
- Add tests for any new functionality.
- Update the relevant documentation page(s) in `DifferentialGames.jl/docs/src/`.
- Summarize the change in the PR description.

## License

DifferentialGames.jl is released under the MIT License. By contributing, you agree that your contributions will be licensed under the same terms.
