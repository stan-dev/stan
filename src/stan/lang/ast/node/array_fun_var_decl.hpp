#ifndef STAN_LANG_AST_NODE_ARRAY_FUN_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_ARRAY_FUN_VAR_DECL_HPP

#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * An array function argument variable declaration.
     */
    struct array_fun_var_decl : public var_decl {
      /**
       * Array type defines element type, array size.
       */
      array_bare_type type_;

      /**
       * Construct an array fun variable declaration with default
       * values. 
       */
      array_fun_var_decl();

      /**
       * Construct an array variable declaration with the specified
       * name and type.
       *
       * @param name variable name
       * @param el_type array element type
       */
      array_fun_var_decl(
                   const std::string& name,
                   const bare_expr_type& el_type);
    };
  }
}
#endif
