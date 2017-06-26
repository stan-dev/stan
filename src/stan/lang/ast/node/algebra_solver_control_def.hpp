#ifndef STAN_LANG_AST_NODE_ALGEBRA_SOLVER_CONTROL_DEF_HPP
#define STAN_LANG_AST_NODE_ALGEBRA_SOLVER_CONTROL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    algebra_solver_control::algebra_solver_control() { }

    algebra_solver_control::algebra_solver_control(
                            const std::string& system_function_name,
                            const expression& x,
                            const expression& y,
                            const expression& dat,
                            const expression& dat_int,
                            const expression& rel_tol,
                            const expression& fun_tol,
                            const expression& max_num_steps)
    : system_function_name_(system_function_name),
      x_(x), y_(y), dat_(dat), dat_int_(dat_int),
      rel_tol_(rel_tol), fun_tol_(fun_tol), max_num_steps_(max_num_steps) {
    }

  }
}
#endif
