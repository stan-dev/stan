#ifndef STAN_LANG_AST_NODE_INT_LOCAL_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_INT_LOCAL_VAR_DECL_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * An integer local variable declaration and optional definition.
     */
    struct int_local_var_decl : public var_decl {
      /**
       * Construct an integer local variable declaration with default
       * values. 
       */
      int_local_var_decl();

      /**
       * Construct an integer local variable declaration with the specified
       * name and definition.
       *
       * @param name variable name
       * @param def definition
       */
      int_local_var_decl(const std::string& name,
                         const expression& def);
    };
  }
}
#endif
