# About

**DifferentialGames.jl** is an umbrella package for computing equilibria of N-player differential games in Julia. It reexports

- [`DifferentialGamesBase.jl`](https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl) — problem types, costs, constraints, dynamics, solutions and the solver interface (`GameSolver`, `solve`, `_solve`, `solver_capabilities`);
- [`DifferentialGamesBaseSolvers.jl`](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl) — the solver implementations.

Documentation conventions follow [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/): mathematical objects are described together with the methods available for them, claims are cited (see [References](references.md)), and the [Notation](notation.md) page lists every symbol.

## Authors

Bennet Outland and contributors.

## Acknowledgments

Bennet Outland thanks the Department of War Science, Math, and Research for Transformation (SMART) Scholarship for academic funding.

## Disclosure of Generative AI usage

Generative AI (Claude Sonnet 4.5/4.6) was used in the creation of this library as a programming aid, including guided code generation, assistance with performance optimization, and documentation. All code and documentation has been reviewed by the author(s) for accuracy.

## Citation

If you use this package in academic work, please cite it as indicated in the repository (`CITATION.cff`, to be added with the JOSS submission).
