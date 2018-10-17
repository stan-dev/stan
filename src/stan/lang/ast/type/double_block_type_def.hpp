#ifndef STAN_LANG_AST_DOUBLE_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_DOUBLE_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
namespace lang {
double_block_type::double_block_type(const range& bounds, const locscale& ls) : bounds_(bounds),
                                                           ls_(ls) {}

double_block_type::double_block_type(const range& bounds) : bounds_(bounds) {}

double_block_type::double_block_type() : double_block_type(range(), locscale()) {}

range double_block_type::bounds() const { return bounds_; }

locscale double_block_type::ls() const { return ls_; }
}  // namespace lang
}  // namespace stan
#endif
