#ifndef STAN_LANG_AST_ORDERED_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_ORDERED_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {
    ordered_block_type::ordered_block_type() { }
    
    ordered_block_type::ordered_block_type(const expression& K)
      : K_(K) { }

    expression ordered_block_type::K() const { return K_; }
  }
}
#endif
