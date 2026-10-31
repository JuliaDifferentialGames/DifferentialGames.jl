# Solver Page Template

Every solver reference page follows the same ten sections so that capabilities, guarantees and limitations can be compared at a glance. Use this checklist when documenting a new solver (see also the extension guide in the [Solver Interface](../api/solver_interface.md)).

1. **Overview and intended use** — one paragraph: what it computes and when to prefer it.
2. **Supported mathematical problem and equilibrium concept** — problem type(s), `solver_capabilities`, and the concept from [Equilibrium Concepts](../math/equilibrium_concepts.md).
3. **Algorithm formulation** — only the equations needed; link to [Optimality Conditions](../math/optimality_conditions.md) and [Numerical Methods](../math/numerical_methods.md) for shared notation.
4. **Assumptions and theoretical guarantees** — state what is proved (with a citation), what is only observed, and where the implementation differs from the paper.
5. **Minimal working Julia example** — an executed `@example` block that constructs a game, solves it, and checks the result (a residual, a dimension, a bound), not hard-coded floating-point output.
6. **Solver configuration** — a table of every constructor option with its default, generated from the struct, plus the `solve` keywords.
7. **Convergence checks and interpretation** — what `sol.converged` means; which `solver_info` entries are residuals; how to verify a solution independently.
8. **Known limitations and failure cases** — unsupported features, sensitivity, assertions that can fire.
9. **Complexity and performance** — only measured or standard results.
10. **References** — original paper(s); if the implementation differs materially, explain how.

Status of the existing pages:

| Solver | Template coverage |
|--------|-------------------|
| [`FNELQ`](../solvers/fnelq.md), [`iLQGames`](../solvers/ilqgames.md), [`ALGAMES`](../solvers/algames.md), [`SFNELQ`](../solvers/sfnelq.md), [`CCiLQGames`](../solvers/ccilqgames.md) | sections 1–10 present (some content shared across headings) |
| [`LIBR`](../solvers/libr.md), [`YiPavel`](../solvers/yipavel.md), [`InverseLQGames`](../solvers/invlq.md), [`RecedingHorizonNash`](../solvers/rhn.md) | partial: option tables verified against the structs; examples and guarantees not yet re-verified |
