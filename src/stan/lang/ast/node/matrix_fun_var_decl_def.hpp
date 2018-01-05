#ifndef STAN_LANG_AST_NODE_MATRIX_FUN_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_FUN_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    matrix_fun_var_decl::matrix_fun_var_decl() { }

    matrix_fun_var_decl::matrix_fun_var_decl(const std::string& name)
      : var_decl(name, bare_expr_type(matrix_type())), is_data_(false) { }

    matrix_fun_var_decl::matrix_fun_var_decl(const std::string& name,
                                             bool is_data)
      : var_decl(name, bare_expr_type(matrix_type())), is_data_(is_data) { }
  }
}
#endif
