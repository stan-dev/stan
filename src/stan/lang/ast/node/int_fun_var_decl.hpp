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
       * True if argument has "data" qualifier.
       */
      bool is_data_;

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

      /**
       * Construct an integer fun_variable declaration with the specified name
       *
       * and is_data flag.
       *
       * @param name fun variable name
       * @param is_data true if declared data_only
       */
      int_fun_var_decl(const std::string& name, bool is_data);
    };
  }
}
#endif
