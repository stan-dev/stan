#ifndef STAN_LANG_AST_FUN_BLOCK_TYPE_IS_SPECIALIZED_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_BLOCK_TYPE_IS_SPECIALIZED_VIS_DEF_HPP

#include <stan/lang/ast/node/range.hpp>
#include <stan/lang/ast/type/block_array_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_corr_block_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_cov_block_type.hpp>
#include <stan/lang/ast/type/corr_matrix_block_type.hpp>
#include <stan/lang/ast/type/cov_matrix_block_type.hpp>
#include <stan/lang/ast/type/double_block_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_block_type.hpp>
#include <stan/lang/ast/type/matrix_block_type.hpp>
#include <stan/lang/ast/type/ordered_block_type.hpp>
#include <stan/lang/ast/type/positive_ordered_block_type.hpp>
#include <stan/lang/ast/type/row_vector_block_type.hpp>
#include <stan/lang/ast/type/simplex_block_type.hpp>
#include <stan/lang/ast/type/unit_vector_block_type.hpp>
#include <stan/lang/ast/type/vector_block_type.hpp>

namespace stan {
namespace lang {
block_type_is_specialized_vis::block_type_is_specialized_vis() {}

bool block_type_is_specialized_vis::operator()(
    const block_array_type& x) const {
  return x.contains().is_specialized();
}

bool block_type_is_specialized_vis::operator()(
    const cholesky_factor_corr_block_type& x) const {
  return true;
}

bool block_type_is_specialized_vis::operator()(
    const cholesky_factor_cov_block_type& x) const {
  return true;
}

bool block_type_is_specialized_vis::operator()(
    const corr_matrix_block_type& x) const {
  return true;
}

bool block_type_is_specialized_vis::operator()(
    const cov_matrix_block_type& x) const {
  return true;
}

bool block_type_is_specialized_vis::operator()(
    const double_block_type& x) const {
  return false;
}

bool block_type_is_specialized_vis::operator()(const ill_formed_type& x) const {
  return false;
}

bool block_type_is_specialized_vis::operator()(const int_block_type& x) const {
  return false;
}

bool block_type_is_specialized_vis::operator()(
    const matrix_block_type& x) const {
  return false;
}

bool block_type_is_specialized_vis::operator()(
    const ordered_block_type& x) const {
  return true;
}

bool block_type_is_specialized_vis::operator()(
    const positive_ordered_block_type& x) const {
  return true;
}

bool block_type_is_specialized_vis::operator()(
    const row_vector_block_type& x) const {
  return false;
}

bool block_type_is_specialized_vis::operator()(
    const simplex_block_type& x) const {
  return true;
}

bool block_type_is_specialized_vis::operator()(
    const unit_vector_block_type& x) const {
  return true;
}

bool block_type_is_specialized_vis::operator()(
    const vector_block_type& x) const {
  return false;
}
}  // namespace lang
}  // namespace stan
#endif
