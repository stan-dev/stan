#ifndef STAN_LANG_AST_NODE_DOUBLE_FUN_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_DOUBLE_FUN_VAR_DECL_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * A double function variable declaration.
     */
    struct double_fun_var_decl : public var_decl {
      /**
       * True if argument has "data" qualifier.
       */
      bool is_data_;

      /**
       * Construct a double fun variable declaration with default
       * values. 
       */
      double_fun_var_decl();

      /**
       * Construct a double fun variable declaration with the specified name.
       *
       * @param name fun variable name
       */
      double_fun_var_decl(const std::string& name);

      /**
       * Construct a double with the specified name and is_data flag.
       *
       * @param name fun variable name
       * @param is_data true if declared data_only
       */
      double_fun_var_decl(const std::string& name, bool is_data);
    };
  }
}
#endif
