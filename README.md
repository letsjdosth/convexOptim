# Convex Optimization

Reference: Boyd, S., Boyd, S. P., & Vandenberghe, L. (2004). Convex optimization. Cambridge university press.

## Unconstrained Minimization

### decent.py

- DescentUnconstrained.run_gradient_descent_with_backtracking_line_search
  - gradient descent method with the backtracking line search.
  - ref: BBV chap 9.2-3. Unconstrained minimization: descent methods, gradient descent method

- DescentUnconstrained.run_steepest_descent_L1_with_backtracking_line_search
- DescentUnconstrained.run_steepest_descent_quadratic_norm_with_backtracking_line_search
  - steepest descent method with the backtracking line search
  - ref: BBV chap 9.4. Unconstrained minimization: steepest descent methods

### newton.py

- NewtonUnconstrained.run_newton_with_backtracking_line_search
  - Newton method with the backtracking line search (using Cholesky decomposition / pseudo inverse)
  - ref: BBV chap 9.5-7. Unconstrained minimization: Newton's method, Self-concordance, Implementation

## Equality Constrained Minimization

### constrained_feasible_start_newton.py

- NewtonAffineConstrainedFeasibleStart.run_newton_with_feasible_starting_point
  - Newton method for equality constraints, with the backtracking line search and a feasible starting point (using Cholesky decomposition / inverse / pseudo inverse)
  - ref: BBV chap 10.2 & 4. Equality constrained minimization: Newton's method with equality constraints, Implementation

### constrained_infeasible_start_newton.py

- NewtonAffineConstrainedInfeasibleStart.run_newton_with_infeasible_starting_point
  - Newton method for equality constraints, with the backtracking line search and any starting point (using Cholesky decomposition / inverse / pseudo inverse)
  - ref: BBV chap 10.3-4. Equality constrained minimization: Infeasible start Newton method, Implementation

## Inequality Constrained Minimization

### barrier.py

- NewtonIneqConstrainedBarrier.outer_run_barrier_method_with_feasible_starting_point
  - Newton method for inequality constraints and affine equality constrains, with the backtracking line search and any starting point (using Cholesky decomposition / inverse / pseudo inverse)
  - ref: BBV chap 11.2-3. Interior-point methods: logarithmic barrier function and central path, the barrier method

- NewtonIneqConstrainedBarrier.outer_run_barrier_method_with_feasible_starting_point
  - Newton method for inequality constraints and affine equality constrains, with the backtracking line search and any starting point (using Cholesky decomposition / inverse / pseudo inverse)
  - ref: BBV chap 11.2-3. Interior-point methods: logarithmic barrier function and central path, the barrier method
  
- Phase1.run_phase1
- Phase1.run_phase1_to_central_path
  - Phase 1 method to find a feasible point under given inequality conditions
  - (11.19) / (11.21) for chapter 11.4.1
  - ref: BBV chap 11.4. Interior-point methods: Feasibility and Phase 1 methods
