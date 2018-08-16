#ifndef STAN_LANG_AST_NODE_VARIABLE_HPP
#define STAN_LANG_AST_NODE_VARIABLE_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <cstddef>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a variable.
     */
    struct variable {
      /**
       * Name of variable.
       */
      std::string name_;

      /**
       * Type of variable.
       */
      expr_type type_;

      /**
       * Construct a default variable.
       */
      variable();

      /**
       * Construct a variable with the specified name and nil type.
       *
       * @param name variable name
       */
      variable(const std::string& name);  // NOLINT(runtime/explicit)

      /**
       * Set the type of the variable to the expression type
       * with the specified base type and number of dimensions.
       *
       * @param base_type base type for variable
       * @param num_dims number of array dims for variable
       */
      void set_type(const base_expr_type& base_type, std::size_t num_dims);
    };

  }
}
#endif
