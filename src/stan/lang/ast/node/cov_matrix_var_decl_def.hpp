#ifndef STAN_LANG_AST_NODE_COV_MATRIX_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_COV_MATRIX_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    cov_matrix_var_decl::cov_matrix_var_decl() : base_var_decl(MATRIX_T) {  }

    cov_matrix_var_decl::cov_matrix_var_decl(expression const& K,
                                         std::string const& name,
                                         std::vector<expression> const& dims,
                                         expression const& def)
      : base_var_decl(name, dims, MATRIX_T, def), K_(K) { }

  }
}
#endif
