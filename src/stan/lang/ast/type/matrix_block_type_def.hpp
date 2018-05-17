#ifndef STAN_LANG_AST_MATRIX_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_MATRIX_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
namespace lang {
matrix_block_type::matrix_block_type(const range& bounds, const expression& M,
                                     const expression& N)
    : bounds_(bounds), M_(M), N_(N) {}

matrix_block_type::matrix_block_type()
    : matrix_block_type(range(), nil(), nil()) {}

range matrix_block_type::bounds() const { return bounds_; }

expression matrix_block_type::M() const { return M_; }

expression matrix_block_type::N() const { return N_; }
}  // namespace lang
}  // namespace stan
#endif
