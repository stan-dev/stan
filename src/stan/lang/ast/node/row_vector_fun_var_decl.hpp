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
       * True if argument has "data" qualifier.
       */
      bool is_data_;

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

      /**
       * Construct a row vector with the specified name and is_data flag.
       *
       * @param name variable name
       * @param is_data true if declared data_only
       */
      row_vector_fun_var_decl(const std::string& name, bool is_data);
    };
  }
}
#endif
