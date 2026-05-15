# Costs & Objectives API

## Stage Costs

```@docs
LQStageCost
```

**Diagonal convenience constructor:** `DiagonalLQStageCost(q_diag, r_diag)` creates an
`LQStageCost` with diagonal `Q = diagm(q_diag)` and `R = diagm(r_diag)`.

## Terminal Costs

```@docs
LQTerminalCost
```

**Diagonal convenience constructor:** `DiagonalLQTerminalCost(qf_diag)` creates an
`LQTerminalCost` with diagonal `Qf = diagm(qf_diag)`.

## Player Objectives

```@docs
PlayerObjective
```

## Custom Cost Functions

All cost types must implement the `AbstractStageCost` or `AbstractTerminalCost` interface
from `DifferentialGamesBase`. The required methods are:

```julia
stage_cost(cost::MyCost, x, u, t) -> Real
terminal_cost(cost::MyCost, x)    -> Real
```

For LQ costs you can use the diagonal convenience constructors:

```julia
stage = DiagonalLQStageCost([1.0, 0.5, 0.1], [0.01])   # 3 states, 1 control
term  = DiagonalLQTerminalCost([10.0, 5.0, 1.0])
```
