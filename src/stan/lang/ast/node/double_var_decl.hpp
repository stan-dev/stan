#ifndef STAN_LANG_AST_NODE_DOUBLE_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_DOUBLE_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * An integer variable declaration and optional definition.
     */
    struct double_var_decl : public base_var_decl {
      /**
       * Range constraint on values with optional lower and upper
       * bounds. 
       */
      range range_;

      /**
       * Construct a real variable declaration with default values.
       */
      double_var_decl();

      /**
       * Construct a real variable declaration with the specified
       * range constraint, name, dimensions, and definition.
       *
       * @param range optional upper and lower bound constraints
       * @param name variable name
       * @param dims array dimensions
       * @param def definition
       */
      double_var_decl(const range& range,
                      const std::string& name,
                      const std::vector<expression>& dims,
                      const expression& def);
    };
  }
}
#endif
