#ifndef STAN_LANG_AST_ROW_VECTOR_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_ROW_VECTOR_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
namespace lang {
row_vector_block_type::row_vector_block_type(const range& bounds,
                                             const expression& N)
    : bounds_(bounds), N_(N) {}

row_vector_block_type::row_vector_block_type()
    : row_vector_block_type(range(), nil()) {}

range row_vector_block_type::bounds() const { return bounds_; }

expression row_vector_block_type::N() const { return N_; }
}  // namespace lang
}  // namespace stan
#endif
