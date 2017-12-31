#ifndef STAN_LANG_AST_NODE_MATRIX_LOCAL_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_LOCAL_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    matrix_local_var_decl::matrix_local_var_decl() { }

    matrix_local_var_decl::matrix_local_var_decl(const std::string& name,
                                                 const expression& M,
                                                 const expression& N)
      : var_decl(name, bare_expr_type(matrix_type())),
        type_(matrix_local_type(M, N)) { }

    matrix_local_var_decl::matrix_local_var_decl(const std::string& name,
                                                 const expression& M,
                                                 const expression& N,
                                                 const expression& def)
      : var_decl(name, bare_expr_type(matrix_type()), def),
        type_(matrix_local_type(M, N)) { }
  }
}
#endif
