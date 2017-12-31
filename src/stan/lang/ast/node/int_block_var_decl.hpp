#ifndef STAN_LANG_AST_NODE_INT_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_INT_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/int_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * An integer block variable declaration and optional definition.
     */
    struct int_block_var_decl : public var_decl {
      /**
       * Type object specifies (optional) lower and upper bounds. 
       */
      int_block_type type_;

      /**
       * Construct an integer block variable declaration with default
       * values. 
       */
      int_block_var_decl();

      /**
       * Construct an integer block variable declaration with the specified
       * name and bounds.
       *
       * @param name variable name
       * @param bounds variable upper and/or lower bounds
       */
      int_block_var_decl(const std::string& name,
                         const range& bounds);

      /**
       * Construct an integer block variable declaration with the specified
       * name, bounds, and definition.
       *
       * @param name variable name
       * @param bounds variable upper and/or lower bounds
       * @param def definition
       */
      int_block_var_decl(const std::string& name,
                         const range& bounds,
                         const expression& def);
    };
  }
}
#endif
