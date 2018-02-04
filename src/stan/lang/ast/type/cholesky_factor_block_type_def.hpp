#ifndef STAN_LANG_AST_CHOLESKY_FACTOR_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_CHOLESKY_FACTOR_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    cholesky_factor_block_type::cholesky_factor_block_type()
      : M_(nil()), N_(nil()) { }

    cholesky_factor_block_type::cholesky_factor_block_type(const expression& M,
                                                           const expression& N)
      : M_(M), N_(N) { }

    expression cholesky_factor_block_type::M() const { return M_; }

    expression cholesky_factor_block_type::N() const { return N_; }
  }
}
#endif
