#ifndef STAN_LANG_AST_NODE_MATRIX_LOCAL_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_LOCAL_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    matrix_local_var_decl::matrix_local_var_decl() { }

    matrix_local_var_decl::matrix_local_var_decl(const std::string& name,
                                                 const matrix_local_type& type)
      : var_decl(name, matrix_type()),
        type_(type.M(), type.N()) { }

    matrix_local_var_decl::matrix_local_var_decl(const std::string& name,
                                                 const matrix_local_type& type,
                                                 const expression& def)
      : var_decl(name, matrix_type(), def),
        type_(type.M(), type.N()) { }
  }
}
#endif
