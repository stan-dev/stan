#ifndef STAN_LANG_AST_NODE_BASE_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_BASE_VAR_DECL_HPP

#include <stan/lang/ast/base_expr_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * AST base class for variable declarations, which share most of
     * their structure.
     */
    struct base_var_decl {
      /**
       * Name of the variable.
       */
      std::string name_;

      /**
       * Dimension sizes for variable.
       */
      std::vector<expression> dims_;

      /**
       * Base type for variable.
       */
      base_expr_type base_type_;

      /**
       * Definition for variable (nil if undefined).
       */
      expression def_;

      /**
       * Construct a default base variable declaration.
       */
      base_var_decl();

      /**
       * Construct a base variable declaration of the specified type.
       *
       * @param base_type base type for variable
       */
      base_var_decl(const base_expr_type& base_type);  // NOLINT

      /**
       * Construct a base variable declaration with the specified
       * name, dimensions, and base type.  The definition is set to
       * nil. 
       *
       * @param name name of variable
       * @param dims dimensions of variables
       * @param base_type base expression type of variable
       */
      base_var_decl(const std::string& name,
                    const std::vector<expression>& dims,
                    const base_expr_type& base_type);

      /**
       * Construct a base variable declaration with the specified
       * name, dimensions, base type, and definition.
       *
       * @param name name of variable
       * @param dims dimensions of variables
       * @param base_type base expression type of variable
       * @param def definition of expression
       */
      base_var_decl(const std::string& name,
                    const std::vector<expression>& dims,
                    const base_expr_type& base_type,
                    const expression& def);
    };

  }
}
#endif
