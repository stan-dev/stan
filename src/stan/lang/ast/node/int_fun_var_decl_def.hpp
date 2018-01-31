#ifndef STAN_LANG_AST_NODE_INT_FUN_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_INT_FUN_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    int_fun_var_decl::int_fun_var_decl() { }

    int_fun_var_decl::int_fun_var_decl(const std::string& name)
      : var_decl(name, int_type()) { }
  }
}
#endif
