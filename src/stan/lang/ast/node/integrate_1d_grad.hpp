#ifndef STAN_LANG_AST_NODE_INTEGRATE_GRAD_1D_HPP
#define STAN_LANG_AST_NODE_INTEGRATE_GRAD_1D_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    struct expression;

    /**
     * Structure for a diff eq integration statement with control
     * parameters for the integrator.
     */
    struct integrate_1d_grad {
      /**
       * The name of the integrator.
       */
      std::string integration_function_name_;

      /**
       * Name of the system 1.
       */
      std::string system_function_1_name_;

      /**
       * Name of the system 2.
       */
      std::string system_function_2_name_;

      /**
       * Lower limit (real).
       */
      expression a_;

      /**
       * Upper limit (real).
       */
      expression b_;

      /**
       * Solution times (real, vector or array of real).
       */
      expression param_;

      /**
       * Construct a default function integrator object with control.
       */
      integrate_1d_grad();

      /**
       * Construt a function integrator with control parameter with the
       * specified values.
       *
       */
      integrate_1d_grad(const std::string& integration_function_name,
                            const std::string& system_function_1_name,
                            const std::string& system_function_2_name,
                            const expression& a,
                            const expression& b,
                            const expression& param);
    };

  }
}
#endif
