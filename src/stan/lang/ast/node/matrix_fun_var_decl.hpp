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
       * True if argument has "data" qualifier.
       */
      bool is_data_;

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

      /**
       * Construct a matrix variable declaration with the specified name
       * and is_data flag.
       *
       * @param name variable name
       * @param is_data true if declared data_only
       */
      matrix_fun_var_decl(const std::string& name, bool is_data);
    };
  }
}
#endif
