#ifndef STAN_LANG_AST_UNIT_VECTOR_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_UNIT_VECTOR_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {
    unit_vector_block_type::unit_vector_block_type() { }

    unit_vector_block_type::unit_vector_block_type(const expression& K)
      : K_(K) { }
  }
}
#endif
