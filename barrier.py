import numpy as np
import scipy.linalg


class NewtonIneqConstrainedBarrier:
    def __init__(self, fn_objective, fn_objective_gradient, fn_objective_hessian, 
                        const_eq_affine_coeff_mat, const_eq_affine_intercept_vec,
                        fn_objective_domain_indicator = None):
       
        self.objective = fn_objective
        self.objective_gradient = fn_objective_gradient
        self.objective_hessian = fn_objective_hessian
        
        if fn_objective_domain_indicator is not None:
            self.objective_domain_indicator = fn_objective_domain_indicator
        else:
            self.objective_domain_indicator = self._Rn_domain_indicator

        self.constraint_eq_coeff_mat = const_eq_affine_coeff_mat
        self.constraint_eq_intercept = const_eq_affine_intercept_vec

        self.constraint_ineq_fn_list = []
        self.constraint_ineq_fn_list_gradient = []
        self.constraint_ineq_fn_list_hessian = []

        self.minimizing_sequence = []
        self.decrement_sequence = []
        self.value_sequence = []

    def set_inequality_constraint(self, fn_negative_constraint, fn_gradient, fn_hessian):
        self.constraint_ineq_fn_list.append(fn_negative_constraint)
        self.constraint_ineq_fn_list_gradient.append(fn_gradient)
        self.constraint_ineq_fn_list_hessian.append(fn_hessian)

    def _Rn_domain_indicator(self, eval_pt):
        return True
    
    def _check_feasible_eq_constraints(self, eval_pt, tolerance):
        eval_intercept = self.constraint_eq_coeff_mat @ eval_pt # matmul
        if abs(eval_intercept - self.constraint_eq_intercept) < tolerance:
            is_feasible = True
        else:
            is_feasible = False
        return is_feasible

    
    def _check_feasible_ineq_constraints(self, eval_pt):
        is_feasible = True
        for f_i in self.constraint_ineq_fn_list:
            if f_i(eval_pt) > 0:
                is_feasible = False
                break
        return is_feasible

    def _logarithmic_barrier(self, eval_pt):
        log_barrier = 0
        for f_i in self.constraint_ineq_fn_list:
            log_barrier -= np.log(f_i(eval_pt)*(-1))
        return log_barrier

    def _logarithmic_barrier_gradient(self, eval_pt):
        dim = eval_pt.size
        log_barrier_gradient = np.zeros(dim)
        for f_i, f1_i in zip(self.constraint_ineq_fn_list, self.constraint_ineq_fn_list_gradient):
            log_barrier_gradient -= f1_i(eval_pt) * (1 / f_i(eval_pt))
        return log_barrier_gradient

    def _logarithmic_barrier_hessian(self, eval_pt):
        dim = eval_pt.size
        log_barrier_hessian = np.zeros((dim, dim))
        for f_i, f1_i, f2_i in zip(self.constraint_ineq_fn_list, self.constraint_ineq_fn_list_gradient, self.constraint_ineq_fn_list_hessian):
            evaluated_f = f_i(eval_pt)
            evaluated_f1 = f1_i(eval_pt)
            part1 = (1/(evaluated_f**2))*(np.outer(evaluated_f1,evaluated_f1))
            part2 = (1/(evaluated_f))*(f2_i(eval_pt))
            log_barrier_hessian += (part1 - part2)
        return log_barrier_hessian

    def _barrier_problem_objective(self, eval_pt, barrier_approx_param_t):
        return self.objective(eval_pt)*barrier_approx_param_t + self._logarithmic_barrier(eval_pt)

    def _barrier_problem_gradient(self, eval_pt, barrier_approx_param_t):
        return self.objective_gradient(eval_pt)*barrier_approx_param_t + self._logarithmic_barrier_gradient(eval_pt)

    def _barrier_problem_hessian(self, eval_pt, barrier_approx_param_t):
        return self.objective_hessian(eval_pt)*barrier_approx_param_t + self._logarithmic_barrier_hessian(eval_pt)
    

    def _get_KKT_matrix_and_intercept(self, eval_pt, barrier_approx_param_t):
        num_row_constraint_eq_coeff_mat = self.constraint_eq_coeff_mat.shape[0]
        KKT_matrix = np.block([
            [self._barrier_problem_hessian(eval_pt, barrier_approx_param_t), self.constraint_eq_coeff_mat.transpose()],
            [self.constraint_eq_coeff_mat,                                   np.zeros((num_row_constraint_eq_coeff_mat, num_row_constraint_eq_coeff_mat))]
            ])
        KKT_intercept = np.block([
            [self._barrier_problem_gradient(eval_pt, barrier_approx_param_t)],
            [np.zeros((1, num_row_constraint_eq_coeff_mat))]
            ]) * (-1)
        return (KKT_matrix, KKT_intercept)


    def _descent_derection_newton_feasible_cholesky(self, eval_pt, barrier_approx_param_t):
        dim = eval_pt.size
        barrier_hessian = self._barrier_problem_hessian(eval_pt, barrier_approx_param_t)
        barrier_gradient = self._barrier_problem_gradient(eval_pt, barrier_approx_param_t)
        
        cholesky_lowtri_of_barrier_hessian = np.linalg.cholesky(barrier_hessian) # L ; H = L(L^T)
        cholesky_lowtri_of_barrier_hessian_inv = scipy.linalg.solve_triangular(cholesky_lowtri_of_barrier_hessian, np.identity(dim), lower=True)
        barrier_hessian_inv = cholesky_lowtri_of_barrier_hessian_inv.transpose() @ cholesky_lowtri_of_barrier_hessian_inv # H^(-1) = (L^(-1))^T L^(-1)
        
        #block elimination
        schur_complement = self.constraint_eq_coeff_mat @ barrier_hessian_inv @ self.constraint_eq_coeff_mat.transpose() * (-1) # S=-AH^(-1)A^T
        
        intercept1 = self.constraint_eq_coeff_mat @ barrier_hessian_inv @ barrier_gradient
        dual_direction = scipy.linalg.solve(schur_complement, intercept1) #w; Sw = AH^(-1)g - h (now, h=0)

        intercept2 = self.constraint_eq_coeff_mat.transpose() @ dual_direction * (-1) - barrier_gradient
        descent_direction = scipy.linalg.solve(barrier_hessian, intercept2) #x; Hx = -A^Tw - g

        newton_decrement_square = sum(descent_direction * barrier_gradient) * (-1)
        return descent_direction, newton_decrement_square
    
    
    def _descent_derection_newton_feasible_inv(self, eval_pt, barrier_approx_param_t):
        KKT_matrix, KKT_intercept = self._get_KKT_matrix_and_intercept(eval_pt, barrier_approx_param_t) # K, k
        KKT_matrix_inv = np.linalg.inv(KKT_matrix)
        sol_KKT_system = np.matmul(KKT_matrix_inv, KKT_intercept) #inverse. K^(-1)k
        descent_direction = (sol_KKT_system[0:eval_pt.size]).flatten()

        newton_decrement_square = sum(KKT_intercept * sol_KKT_system) #[neg_grad, 0], [x, w]
        return descent_direction, newton_decrement_square

    def _descent_derection_newton_feasible_pinv(self, eval_pt, barrier_approx_param_t):
        KKT_matrix, KKT_intercept = self._get_KKT_matrix_and_intercept(eval_pt, barrier_approx_param_t) # K, k
        sol_KKT_system = np.matmul(np.linalg.pinv(KKT_matrix), KKT_intercept) #pseudo. K^(-1)k
        descent_direction = (sol_KKT_system[0:eval_pt.size]).flatten()

        newton_decrement_square = sum(KKT_intercept * sol_KKT_system) #[neg_grad, 0], [x, w]
        return descent_direction, newton_decrement_square


    def _backtracking_line_search(self, eval_pt, barrier_approx_param_t,
            descent_direction, 
            a_slope_flatter_ratio, b_step_shorten_ratio):

        if a_slope_flatter_ratio <= 0 or a_slope_flatter_ratio >= 0.5:
            raise ValueError("a should be 0 < a < 0.5")
        if b_step_shorten_ratio <= 0 or b_step_shorten_ratio >= 1:
            raise ValueError("b should be 0 < a < 1")

        step_size = 1

        while True:
            flatten_line_slope = self._barrier_problem_gradient(eval_pt, barrier_approx_param_t) * a_slope_flatter_ratio * step_size
            deviation_vec = descent_direction * step_size

            objective_fn_value = self._barrier_problem_objective(eval_pt + deviation_vec, barrier_approx_param_t)
            flatten_line_value = self._barrier_problem_objective(eval_pt, barrier_approx_param_t) + sum(flatten_line_slope * deviation_vec)

            if objective_fn_value < flatten_line_value and self.objective_domain_indicator(eval_pt + deviation_vec):
                break
            else:
                step_size = step_size * b_step_shorten_ratio

        return step_size


    def _inner_run_newton_with_feasible_starting_point(self, starting_pt, 
                                                barrier_approx_param_t,
                                                tolerance, 
                                                a_slope_flatter_ratio, b_step_shorten_ratio,
                                                method="cholesky"):
        #method : cholesky, inv, pinv (if hessian is singular)
        is_feasible = self._check_feasible_eq_constraints(starting_pt, tolerance = 0.0000001)
        if not is_feasible:
            raise ValueError("not feasible starting point (w.r.t. equality constraints).")


        num_inner_iter = 0
        while True:
            eval_pt = self.minimizing_sequence[-1][-1]
            if method == "cholesky":
                descent_direction, decrement_square = self._descent_derection_newton_feasible_cholesky(eval_pt, barrier_approx_param_t)
            elif method == "inv":
                descent_direction, decrement_square = self._descent_derection_newton_feasible_inv(eval_pt, barrier_approx_param_t)
            elif method == "pinv":
                descent_direction, decrement_square = self._descent_derection_newton_feasible_pinv(eval_pt, barrier_approx_param_t)
            else:
                raise ValueError("method should be ['cholesky', 'inv', 'pinv']")
            decrement = np.sqrt(decrement_square)
            self.decrement_sequence[-1].append(decrement)

            if decrement_square < (tolerance*2):
                break

            descent_step_size = self._backtracking_line_search(eval_pt, barrier_approx_param_t,
                                    descent_direction, 
                                    a_slope_flatter_ratio, b_step_shorten_ratio)
            next_point = eval_pt + descent_direction * descent_step_size

            self.minimizing_sequence[-1].append(next_point)
            self.value_sequence[-1].append(self.objective(next_point))
            num_inner_iter += 1

        print("inner iteration:", num_inner_iter)

    def outer_run_barrier_method_with_feasible_starting_point(self, 
                        starting_pt, 
                        t0_starting_barrier_approx_param,
                        mu_multiplicative_approx_factor,
                        outer_iteration_tolerance,
                        inner_iteration_tolerance=0.001,
                        inner_method="cholesky",
                        a_slope_flatter_ratio=0.2,
                        b_step_shorten_ratio=0.5):
        
        now_eval_pt = starting_pt
        
        #start with phase 1 method here
        #now, not implemented, so I will raise an error
        is_ineq_feasible = self._check_feasible_ineq_constraints(now_eval_pt)
        if is_ineq_feasible is False:
            raise ValueError("not feasible starting point (w.r.t. inequality constraints)")

        #phase 2
        num_outer_iter = 0
        barrier_approx_param_t = t0_starting_barrier_approx_param
        num_ineq_constraints = len(self.constraint_ineq_fn_list)
        while True:
            num_outer_iter += 1
            self.minimizing_sequence.append([now_eval_pt])
            self.value_sequence.append([self.objective(now_eval_pt)])
            self.decrement_sequence.append([])

            self._inner_run_newton_with_feasible_starting_point(now_eval_pt, 
                                                barrier_approx_param_t,
                                                inner_iteration_tolerance, 
                                                a_slope_flatter_ratio, b_step_shorten_ratio,
                                                method=inner_method)
            
            now_eval_pt = self.minimizing_sequence[-1][-1]

            #stopping
            if num_ineq_constraints/barrier_approx_param_t < outer_iteration_tolerance:
                break

            barrier_approx_param_t *= mu_multiplicative_approx_factor

        print("outer iteration:", num_outer_iter)


    def get_minimizing_sequence(self):
        return self.minimizing_sequence
    
    def get_minimizing_function_value_sequence(self):
        return self.value_sequence

    def get_decrement_sequence(self):
        return self.decrement_sequence

    def get_arg_min(self):
        return self.minimizing_sequence[-1][-1]

    def get_min(self):
        return self.objective(self.minimizing_sequence[-1][-1])



if __name__=="__main__":
    #x*=2, p*=5
    def test1_objective(x):
        return x**2+1
    def test1_objective_gradient(x):
        return np.array([2*x])
    def test1_objective_hessian(x):
        return np.array([[2]])
    
    def test1_ineq_const1(x):
        return -x+2
    def test1_ineq_const1_gradient(x):
        return np.array([-1])
    def test1_ineq_const1_hessian(x):
        return np.array([[0]])

    def test1_ineq_const2(x):
        return x-4
    def test1_ineq_const2_gradient(x):
        return np.array([1])
    def test1_ineq_const2_hessian(x):
        return np.array([[0]])



    #x*=(2,2), p*=6
    def test2_objective(vec_2dim):
        x, y = vec_2dim
        return x**2 + (y-1)**2 + 1
    def test2_objective_gradient(vec_2dim):
        x, y = vec_2dim
        return np.array([2*x, 2*y-2])
    def test2_objective_hessian(vec_2dim):
        return np.array([[2,0],
                        [0,2]])
    
    def test2_ineq_const1(vec_2dim):
        x, y = vec_2dim
        return -x+2
    def test2_ineq_const1_gradient(vec_2dim):
        return np.array([-1, 0])
    def test2_ineq_const1_hessian(vec_2dim):
        return np.array([[0, 0],
                        [0, 0]])

    def test2_ineq_const2(vec_2dim):
        x, y = vec_2dim
        return x-4
    def test2_ineq_const2_gradient(vec_2dim):
        return np.array([1, 0])
    def test2_ineq_const2_hessian(vec_2dim):
        return np.array([[0, 0],
                        [0, 0]])
    test2_A = np.array([[1, -1]])
    test2_b = np.array([0])

    
    test_inst_2 = NewtonIneqConstrainedBarrier(test2_objective, test2_objective_gradient, test2_objective_hessian, test2_A, test2_b)
    test_inst_2.set_inequality_constraint(test2_ineq_const1, test2_ineq_const1_gradient, test2_ineq_const1_hessian)
    test_inst_2.set_inequality_constraint(test2_ineq_const2, test2_ineq_const2_gradient, test2_ineq_const2_hessian)
    test_inst_2.outer_run_barrier_method_with_feasible_starting_point(np.array([3, 3]), 1, 10, 0.01)
    print(test_inst_2.get_minimizing_sequence())
    print(test_inst_2.get_minimizing_function_value_sequence())


    #x*=(2,1), p*=9
    def test3_objective(vec_2dim):
        x, y = vec_2dim
        return (x-y)**2 + 2*(x**2)
    def test3_objective_gradient(vec_2dim):
        x, y = vec_2dim
        return np.array([6*x-2*y, 2*y-2*x])
    def test3_objective_hessian(vec_2dim):
        return np.array([[6,-2],
                        [-2,2]])
    
    def test3_ineq_const1(vec_2dim):
        x, y = vec_2dim
        return -x+2
    def test3_ineq_const1_gradient(vec_2dim):
        return np.array([-1, 0])
    def test3_ineq_const1_hessian(vec_2dim):
        return np.array([[0, 0],
                        [0, 0]])
    test3_A = np.array([[1, -2]])
    test3_b = np.array([0])

    test_inst_3 = NewtonIneqConstrainedBarrier(test3_objective, test3_objective_gradient, test3_objective_hessian, test3_A, test3_b)
    test_inst_3.set_inequality_constraint(test3_ineq_const1, test3_ineq_const1_gradient, test3_ineq_const1_hessian)
    test_inst_3.outer_run_barrier_method_with_feasible_starting_point(np.array([10, 5]), 1, 10, 0.01)
    print(test_inst_3.get_minimizing_sequence())
    print(test_inst_3.get_minimizing_function_value_sequence())

