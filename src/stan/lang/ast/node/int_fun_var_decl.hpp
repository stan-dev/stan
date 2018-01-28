#ifndef STAN_LANG_AST_NODE_INT_FUN_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_INT_FUN_VAR_DECL_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * An integer function variable declaration and optional definition.
     */
    struct int_fun_var_decl : public var_decl {
      /**
       * Construct an integer fun_variable declaration with default
       * values. 
       */
      int_fun_var_decl();

      /**
       * Construct an integer fun_variable declaration with the specified name.
       *
       * @param name fun_variable name
       */
      int_fun_var_decl(const std::string& name);

    };
  }
}
#endif
