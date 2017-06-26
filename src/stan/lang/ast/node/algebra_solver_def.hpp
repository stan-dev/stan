#ifndef STAN_LANG_AST_NODE_ALGEBRA_SOLVER_DEF_HPP
#define STAN_LANG_AST_NODE_ALGEBRA_SOLVER_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    algebra_solver::algebra_solver() { }

    algebra_solver::algebra_solver(const std::string& system_function_name,
                                   const expression& x,
                                   const expression& y,
                                   const expression& dat,
                                   const expression& dat_int)
      : system_function_name_(system_function_name),
        x_(x), y_(y), dat_(dat), dat_int_(dat_int) { }

    }
}

#endif
