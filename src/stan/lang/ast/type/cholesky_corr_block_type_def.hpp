#ifndef STAN_LANG_AST_CHOLESKY_CORR_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_CHOLESKY_CORR_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    cholesky_corr_block_type::cholesky_corr_block_type() { }

    cholesky_corr_block_type::cholesky_corr_block_type(const expression& K)
      : K_(K) { }

    expression cholesky_corr_block_type::K() const { return K_; }
  }
}
#endif
