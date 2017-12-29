#ifndef STAN_LANG_AST_CHOLESKY_FACTOR_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_CHOLESKY_FACTOR_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    cholesky_factor_block_type::cholesky_factor_block_type() { }

    cholesky_factor_block_type::cholesky_factor_block_type(const expression& M,
                                                           const expression& N)
      : M_(M), N_(N) { }
  }
}
#endif
