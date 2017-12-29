#ifndef STAN_LANG_AST_MATRIX_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_MATRIX_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {
    matrix_block_type::matrix_block_type() { }

    matrix_block_type::matrix_block_type(const range& bounds,
                                         const expression& M,
                                         const expression& N)
      : bounds_(bounds), M_(M), N_(N) { }
  }
}
#endif
