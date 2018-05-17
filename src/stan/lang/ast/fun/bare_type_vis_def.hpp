#ifndef STAN_LANG_AST_FUN_BARE_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_BARE_TYPE_VIS_DEF_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/type/block_array_type.hpp>
#include <stan/lang/ast/type/local_array_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_corr_block_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_cov_block_type.hpp>
#include <stan/lang/ast/type/corr_matrix_block_type.hpp>
#include <stan/lang/ast/type/cov_matrix_block_type.hpp>
#include <stan/lang/ast/type/double_block_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_block_type.hpp>
#include <stan/lang/ast/type/int_type.hpp>
#include <stan/lang/ast/type/matrix_block_type.hpp>
#include <stan/lang/ast/type/matrix_local_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/ordered_block_type.hpp>
#include <stan/lang/ast/type/positive_ordered_block_type.hpp>
#include <stan/lang/ast/type/row_vector_block_type.hpp>
#include <stan/lang/ast/type/row_vector_local_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/simplex_block_type.hpp>
#include <stan/lang/ast/type/unit_vector_block_type.hpp>
#include <stan/lang/ast/type/vector_block_type.hpp>
#include <stan/lang/ast/type/vector_local_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>

namespace stan {
namespace lang {
bare_type_vis::bare_type_vis() {}
  
bare_expr_type bare_type_vis::operator()(const block_array_type& x) const {
  return bare_array_type(x.contains().bare_type(), x.dims());
}

bare_expr_type bare_type_vis::operator()(const local_array_type& x) const {
  return bare_array_type(x.contains().bare_type(), x.dims());
}

bare_expr_type bare_type_vis::operator()(
    const cholesky_factor_corr_block_type& x) const {
  return matrix_type();
}

bare_expr_type bare_type_vis::operator()(
    const cholesky_factor_cov_block_type& x) const {
  return matrix_type();
}

bare_expr_type bare_type_vis::operator()(
    const corr_matrix_block_type& x) const {
  return matrix_type();
}

bare_expr_type bare_type_vis::operator()(const cov_matrix_block_type& x) const {
  return matrix_type();
}

bare_expr_type bare_type_vis::operator()(const double_block_type& x) const {
  return double_type();
}

bare_expr_type bare_type_vis::operator()(const double_type& x) const {
  return double_type();
}

bare_expr_type bare_type_vis::operator()(const ill_formed_type& x) const {
  return ill_formed_type();
}

bare_expr_type bare_type_vis::operator()(const int_block_type& x) const {
  return int_type();
}

bare_expr_type bare_type_vis::operator()(const int_type& x) const {
  return int_type();
}

bare_expr_type bare_type_vis::operator()(const matrix_block_type& x) const {
  return matrix_type();
}

bare_expr_type bare_type_vis::operator()(const matrix_local_type& x) const {
  return matrix_type();
}

bare_expr_type bare_type_vis::operator()(const ordered_block_type& x) const {
  return vector_type();
}

bare_expr_type bare_type_vis::operator()(
    const positive_ordered_block_type& x) const {
  return vector_type();
}

bare_expr_type bare_type_vis::operator()(const row_vector_block_type& x) const {
  return row_vector_type();
}

bare_expr_type bare_type_vis::operator()(const row_vector_local_type& x) const {
  return row_vector_type();
}

bare_expr_type bare_type_vis::operator()(const simplex_block_type& x) const {
  return vector_type();
}

bare_expr_type bare_type_vis::operator()(
    const unit_vector_block_type& x) const {
  return vector_type();
}

bare_expr_type bare_type_vis::operator()(const vector_block_type& x) const {
  return vector_type();
}

bare_expr_type bare_type_vis::operator()(const vector_local_type& x) const {
  return vector_type();
}
}  // namespace lang
}  // namespace stan
#endif
