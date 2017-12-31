#ifndef STAN_LANG_AST_NODE_DOUBLE_LOCAL_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_DOUBLE_LOCAL_VAR_DECL_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * A double local variable declaration and optional definition.
     */
    struct double_local_var_decl : public var_decl {
      /**
       * Construct a double local variable declaration with default
       * values. 
       */
      double_local_var_decl();

      /**
       * Construct a double local variable declaration with the specified name.n
       *
       * @param name variable name
       */
      double_local_var_decl(const std::string& name);

      /**
       * Construct a double local variable declaration with the specified
       * name and definition.
       *
       * @param name variable name
       * @param def definition
       */
      double_local_var_decl(const std::string& name,
                            const expression& def);
    };
  }
}
#endif
