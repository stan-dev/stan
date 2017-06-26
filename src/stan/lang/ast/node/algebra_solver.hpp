#ifndef STAN_LANG_AST_NODE_ALGEBRA_SOLVER_HPP
#define STAN_LANG_AST_NODE_ALGEBRA_SOLVER_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    struct expression;

    /**
     * Structure for algebraic solver statement.
     */
    struct algebra_solver {
      /**
       * Name of the algebra system.
       */
      std::string system_function_name_;

      /**
       * Initial guess for solution.
       */
      expression x_;

      /**
       * Parameters.
       */
      expression y_;

      /**
       * Real-valued data.
       */
      expression dat_;

      /**
       * Integer-valued data.
       */
      expression dat_int_;

      /**
       * Construct a default algebra solver node.
       */
      algebra_solver();

      /**
       * Construct an algebraic solver.
       *
       * @param system_function_name name of ODE system
       * @param x initial guess for solution
       * @param y parameters
       * @param dat real-valued data
       * @param dat_int integer-valued data
       */
      algebra_solver(const std::string& system_function_name,
                     const expression& x,
                     const expression& y,
                     const expression& dat,
                     const expression& dat_int);
    };

  }
}
#endif
