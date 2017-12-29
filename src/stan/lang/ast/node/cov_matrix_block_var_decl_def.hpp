#ifndef STAN_LANG_AST_NODE_COV_MATRIX_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_COV_MATRIX_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    cov_matrix_block_var_decl::cov_matrix_block_var_decl() { }

    cov_matrix_block_var_decl::cov_matrix_block_var_decl(
                               const std::string& name,
                               const expression& K,
                               const expression& def)
      : var_decl(name, bare_expr_type(matrix_type()), def),
        type_(cov_matrix_block_type(K)) { }
  }
}
#endif
