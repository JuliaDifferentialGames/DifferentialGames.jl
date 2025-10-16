# DifferentialGamesBase

<!--
[![Build Status](https://github.com/BennetOutland/DifferentialGamesBase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BennetOutland/DifferentialGamesBase.jl/actions/workflows/CI.yml?query=branch%3Amain)
-->

[![CI](https://github.com/JuliaDifferentialGames/DifferentialGamesBase/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDifferentialGames/DifferentialGamesBase/actions/workflows/CI.yml)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

## Description

This is a work in progress base package for solving various different types of differential games. The goal is to first support the following types of differential game problem types:

- Zero-sum Games
- LQ Games
- Deterministic General Nash Equilibrium Problems
- Stochastic General Nash Equilibrium Problems
- Jump Diffusion General Nash Equilibrium Problems
- Inverse General Nash Equilibrium Problems

Afterwards, the goal is to support:

- Mean field games
- Games with heterogenous players
- Games with learned objective functions


## Strucuture Inspiration:

- https://github.com/SciML/SciMLBase.jl/tree/master/src
- https://github.com/SciML/DiffEqBase.jl/tree/master/src
- https://github.com/SciML/Optimization.jl/tree/master/src  + https://github.com/robertfeldt/BlackBoxOptim.jl
