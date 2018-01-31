#ifndef STAN_LANG_AST_NODE_CHOLESKY_CORR_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_CHOLESKY_CORR_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    cholesky_corr_block_var_decl::cholesky_corr_block_var_decl() { }

    cholesky_corr_block_var_decl::cholesky_corr_block_var_decl(
                                  const std::string& name,
                                  const cholesky_corr_block_type& type)
      : var_decl(name, matrix_type()), type_(type.K()) { }

    cholesky_corr_block_var_decl::cholesky_corr_block_var_decl(
                                  const std::string& name,
                                  const cholesky_corr_block_type& type,
                                  const expression& def)
      : var_decl(name, matrix_type(), def), type_(type.K()) { }
  }
}
#endif
