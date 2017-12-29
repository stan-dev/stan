#ifndef STAN_LANG_AST_NODE_ARRAY_FUN_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ARRAY_FUN_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    array_fun_var_decl::array_fun_var_decl() { }

    array_fun_var_decl::array_fun_var_decl(
                          const std::string& name,
                          const bare_expr_type& el_type)
      : var_decl(name), type_(array_bare_type(el_type)) { }
  }
}
#endif
