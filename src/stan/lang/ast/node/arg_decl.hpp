#ifndef STAN_LANG_AST_NODE_ARG_DECL_HPP
#define STAN_LANG_AST_NODE_ARG_DECL_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/node/base_var_decl.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * AST node for the type delclaration for function arguments.
     */
    struct arg_decl {
      /**
       * Construct an uninitialized argument declaration.
       */
      arg_decl();

      /**
       * Construct an argument declaration with the specified type and
       * variable name.
       *
       * @param arg_type argument variable type
       * @param name argument variable name
       */
      arg_decl(const expr_type& arg_type, const std::string& name);

      /**
       * Return the base declaration corresponding to this argument
       * declaration. 
       *
       * @return variable declaration for this argument
       */
      base_var_decl base_variable_declaration() const;

      /**
       * Type of the argument variable.
       */
      expr_type arg_type_;

      /**
       * Name of the argument variable.
       */
      std::string name_;
    };

  }
}
#endif
