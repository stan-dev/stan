#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_FUN_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_FUN_VAR_DECL_HPP

#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a row vector function argument variable declaration.
     */
    struct row_vector_fun_var_decl : public var_decl {
      /**
       * Construct a row vector variable declaration with default
       * values.
       */
      row_vector_fun_var_decl();

      /**
       * Construct a row vector with the specified name.
       *
       * @param name variable name
       */
      row_vector_fun_var_decl(const std::string& name);
    };
  }
}
#endif
