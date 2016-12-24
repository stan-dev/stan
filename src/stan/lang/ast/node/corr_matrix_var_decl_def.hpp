#ifndef STAN_LANG_AST_NODE_CORR_MATRIX_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_CORR_MATRIX_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    corr_matrix_var_decl::corr_matrix_var_decl() : base_var_decl(MATRIX_T) { }

    corr_matrix_var_decl::corr_matrix_var_decl(const expression& K,
                                   const std::string& name,
                                   const std::vector<expression>& dims,
                                   const expression& def)

      : base_var_decl(name, dims, MATRIX_T, def), K_(K) { }

  }
}
#endif
