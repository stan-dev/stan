#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_LOCAL_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_LOCAL_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    row_vector_local_var_decl::row_vector_local_var_decl() { }

    row_vector_local_var_decl::row_vector_local_var_decl(
                               const std::string& name,
                               const row_vector_local_type& type)
      : var_decl(name, row_vector_type()), type_(type.N()) { }

    row_vector_local_var_decl::row_vector_local_var_decl(
                               const std::string& name,
                               const row_vector_local_type& type,
                               const expression& def)
      : var_decl(name, row_vector_type(), def), type_(type.N()) { }
  }
}
#endif
