#ifndef STAN_LANG_AST_NODE_MATRIX_FUN_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_MATRIX_FUN_VAR_DECL_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a matrix fun variable declaration.
     */
    struct matrix_fun_var_decl : public var_decl {
      /**
       * Construct a matrix fun variable declaration with default values.
       */
      matrix_fun_var_decl();

      /**
       * Construct a matrix variable declaration with the specified name.
       *
       * @param name variable name
       */
      matrix_fun_var_decl(const std::string& name);
    };
  }
}
#endif
