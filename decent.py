import numpy as np


class DescentUnconstrainted:
    def __init__(self, fn_objective, fn_objective_gradient, 
        fn_objective_hessian = None, fn_objective_domain_indicator = None):
        
        self.objective = fn_objective
        self.objective_gradient = fn_objective_gradient
        if fn_objective_domain_indicator is not None:
            self.objective_domain_indicator = fn_objective_domain_indicator
        else:
            self.objective_domain_indicator = self._Rn_domain_indicator

    def _Rn_domain_indicator(self, eval_pt):
        return True

    def _backtracking_line_search(self, eval_pt, descent_direction, 
            a_slope_flatter_ratio, b_step_shorten_ratio):

        if a_slope_flatter_ratio <= 0 or a_slope_flatter_ratio >= 0.5:
            raise ValueError("a should be 0 < a < 0.5")
        if b_step_shorten_ratio <= 0 or b_step_shorten_ratio >= 1:
            raise ValueError("b should be 0 < a < 1")

        step_size = 1

        while True:
            flatten_line_slope = self.objective_gradient(eval_pt) * a_slope_flatter_ratio * step_size
            deviation_vec = descent_direction * step_size

            objective_fn_value = self.objective(eval_pt + deviation_vec)
            flatten_line_value = self.objective(eval_pt) + sum(flatten_line_slope * deviation_vec)

            if objective_fn_value < flatten_line_value and self.objective_domain_indicator(eval_pt + deviation_vec):
                break
            else:
                step_size = step_size * b_step_shorten_ratio

        return step_size
    
    def _l2_norm(self, vec):
        return (sum(vec**2))**0.5
    
    def run_gradient_descent_with_backtracking_line_search(self, starting_pt, tolerance = 0.01):
        self.minimizing_sequence = [starting_pt]
        self.value_sequence = [self.objective(starting_pt)]
        num_iter = 0
        while True:
            eval_pt = self.minimizing_sequence[-1]
            descent_direction = self.objective_gradient(eval_pt) * (-1)
            if self._l2_norm(descent_direction) < tolerance:
                break
            descent_step_size = self._backtracking_line_search(eval_pt, descent_direction, 
                                    a_slope_flatter_ratio = 0.2, b_step_shorten_ratio = 0.5)
            next_point = eval_pt + descent_direction * descent_step_size
            self.minimizing_sequence.append(next_point)
            self.value_sequence.append(self.objective(next_point))
            num_iter += 1

        print("iteration: ", num_iter)

    def get_minimizing_sequence(self):
        return self.minimizing_sequence
    
    def get_minimizing_function_value_sequence(self):
        return self.value_sequence


    def get_arg_min(self):
        return self.minimizing_sequence[-1]

    def get_min(self):
        return self.objective(self.minimizing_sequence[-1])


if __name__ == "__main__":
    
    #test 1
    def test_objective1(vec_2dim, gamma = 2):
        val = 0.5 * (vec_2dim[0]**2 + gamma * (vec_2dim[1]**2))
        return np.array(val)

    def test_objective1_gradient(vec_2dim, gamma = 2):
        grad = (vec_2dim[0], vec_2dim[1] * gamma)
        return np.array(grad)


    test_descent_inst = DescentUnconstrainted(test_objective1, test_objective1_gradient)
    test_descent_inst.run_gradient_descent_with_backtracking_line_search(np.array([13,22.3]))
    print(test_descent_inst.get_minimizing_sequence())
    print(test_descent_inst.get_minimizing_function_value_sequence())

    #test 2
    def test_objective2(vec_2dim, gamma = 2):
        val = np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) + np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1) + np.exp(-vec_2dim[0] - 0.1)
        return np.array(val)

    def test_objective2_gradient(vec_2dim, gamma = 2):
        grad = (np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) + np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1) - np.exp(-vec_2dim[0] - 0.1),
                3 * np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) - 3 * np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1))
        return np.array(grad)


    test_descent_inst2 = DescentUnconstrainted(test_objective2, test_objective2_gradient)
    test_descent_inst2.run_gradient_descent_with_backtracking_line_search(np.array([9, 20]))
    print(test_descent_inst2.get_minimizing_sequence())
    print(test_descent_inst2.get_minimizing_function_value_sequence())


