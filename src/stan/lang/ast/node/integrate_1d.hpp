#ifndef STAN_LANG_AST_NODE_INTEGRATE_1D_HPP
#define STAN_LANG_AST_NODE_INTEGRATE_1D_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    struct expression;

    /**
     * Structure for a diff eq integration statement with control
     * parameters for the integrator.
     */
    struct integrate_1d {
      /**
       * The name of the integrator.
       */
      std::string integration_function_name_;

      /**
       * Name of the system.
       */
      std::string system_function_name_;

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
      integrate_1d();

      /**
       * Construt a function integrator with control parameter with the
       * specified values.
       *
       */
      integrate_1d(const std::string& integration_function_name,
                            const std::string& system_function_name,
                            const expression& a,
                            const expression& b,
                            const expression& param);
    };

  }
}
#endif
