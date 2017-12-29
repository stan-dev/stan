#ifndef STAN_LANG_AST_NODE_DOUBLE_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_DOUBLE_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/double_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * A double block variable declaration and optional definition.
     */
    struct double_block_var_decl : public var_decl {
      /**
       * Type object specifies (optional) lower and upper bounds. 
       */
      double_block_type type_;

      /**
       * Construct a double block variable declaration with default
       * values. 
       */
      double_block_var_decl();

      /**
       * Construct a double block variable declaration with the specified
       * name, bounds, and definition.
       *
       * @param name variable name
       * @param bounds variable upper and/or lower bounds
       * @param def definition
       */
      double_block_var_decl(const std::string& name,
                            const range& bounds,
                            const expression& def);
    };
  }
}
#endif
