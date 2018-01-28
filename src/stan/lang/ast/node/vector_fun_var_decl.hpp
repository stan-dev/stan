#ifndef STAN_LANG_AST_NODE_VECTOR_FUN_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_VECTOR_FUN_VAR_DECL_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a column vector function argument variable declaration.
     */
    struct vector_fun_var_decl : public var_decl {
      /**
       * Construct a column vector variable declaration with default
       * values.
       */
      vector_fun_var_decl();

      /**
       * Construct a column vector with the specified name.
       *
       * @param name variable name
       */
      vector_fun_var_decl(const std::string& name);
    };
  }
}
#endif
