#ifndef STAN_MODEL_INDEXING_ASSIGN_CL_HPP
#define STAN_MODEL_INDEXING_ASSIGN_CL_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/rev.hpp>
#include <stan/math/opencl/indexing_rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/rvalue_cl.hpp>
#include <utility>

namespace stan {
namespace model {

namespace internal {
inline constexpr const char* print_index_type(stan::model::index_uni) {
  return "uni index";
}

inline const char* print_index_type(stan::model::index_multi) {
  return "multi index";
}

inline constexpr const char* print_index_type(stan::model::index_min) {
  return "min index";
}

inline constexpr const char* print_index_type(stan::model::index_max) {
  return "max index";
}

inline constexpr const char* print_index_type(stan::model::index_min_max) {
  return "min max index";
}

inline constexpr const char* print_index_type(stan::model::index_omni) {
  return "omni index";
}

#ifdef STAN_OPENCL
inline const char* print_index_type(const stan::math::matrix_cl<int>&) {
  return "multi index";
}
#endif
}  // namespace internal

// prim
/**
 * Assign one primitive kernel generator expression to another, using given
 * index.
 *
 * @tparam ExprLhs type of the assignable prim expression on the left hand side
 * of the assignment
 * @tparam ExprRhs type of the prim expression on the right hand side of the
 * assignment
 * @tparam RowIndex type of index (a Stan index type or `matrix_cl<int>` instead
 * of `index_multi`)
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param expr_rhs expression on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 * @throw std::invalid_argument If the right hand side size isn't the same as
 * the indexed left hand side size.
 */
template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          require_kernel_expression_lhs_t<ExprLhs>* = nullptr,
          require_all_kernel_expressions_and_none_scalar_t<ExprRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, ExprRhs&& expr_rhs, const char* name,
                   const RowIndex& row_index) {
  decltype(auto) lhs = rvalue(expr_lhs, name, row_index);
  stan::math::check_size_match(internal::print_index_type(row_index),
                               "left hand side rows", lhs.rows(), name,
                               expr_rhs.rows());
  stan::math::check_size_match(internal::print_index_type(row_index),
                               "left hand side cols", lhs.cols(), name,
                               expr_rhs.cols());
  lhs = std::forward<ExprRhs>(expr_rhs);
}

/**
 * Assign one primitive kernel generator expression to another, using given
 * indices.
 *
 * @tparam ExprLhs type of the assignable prim expression on the left hand side
 * of the assignment
 * @tparam ExprRhs type of the prim expression on the right hand side of the
 * assignment
 * @tparam RowIndex type of row index (a Stan index type or `matrix_cl<int>`
 * instead of `index_multi`)
 * @tparam ColIndex type of column index (a Stan index type or `matrix_cl<int>`
 * instead of `index_multi`)
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param expr_rhs expression on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing rows of `expr_lhs`
 * @param col_index index used for indexing columns of `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 * @throw std::invalid_argument If the right hand side size isn't the same as
 * the indexed left hand side size.
 */
template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          typename ColIndex,
          require_kernel_expression_lhs_t<ExprLhs>* = nullptr,
          require_all_kernel_expressions_and_none_scalar_t<ExprRhs>* = nullptr,
          require_any_not_same_t<RowIndex, ColIndex, index_uni>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, ExprRhs&& expr_rhs, const char* name,
                   const RowIndex& row_index, const ColIndex& col_index) {
  decltype(auto) lhs = rvalue(expr_lhs, name, row_index, col_index);
  stan::math::check_size_match(internal::print_index_type(row_index),
                               "left hand side rows", lhs.rows(), name,
                               expr_rhs.rows());
  stan::math::check_size_match(internal::print_index_type(col_index),
                               "left hand side cols", lhs.cols(), name,
                               expr_rhs.cols());

  lhs = std::forward<ExprRhs>(expr_rhs);
}

/**
 * Assign a scalar to a primitive kernel generator expression, using given uni
 * indices.
 *
 * @tparam ExprLhs type of the assignable prim expression on the left hand side
 * of the assignment
 * @tparam ScalRhs type of the prim scalar on the right hand side of the
 * assignment
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param scal_rhs scalar on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing rows of `expr_lhs`
 * @param col_index index used for indexing columns of `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename ExprLhs, typename ScalRhs,
          require_kernel_expression_lhs_t<ExprLhs>* = nullptr,
          require_stan_scalar_t<ScalRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ScalRhs& scal_rhs,
                   const char* name, const index_uni row_index,
                   const index_uni col_index) {
  math::block_zero_based(expr_lhs, row_index.n_ - 1, col_index.n_ - 1, 1, 1)
      = math::constant(scal_rhs, 1, 1);
}

// rev
/**
 * Assign one primitive or reverse mode kernel generator expression to a reverse
 * mode one, using given index.
 *
 * @tparam ExprLhs type of the assignable rev expression on the left hand side
 * of the assignment
 * @tparam ExprRhs type of the prim or rev expression on the right hand side of
 * the assignment
 * @tparam RowIndex type of index
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param expr_rhs expression on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing `expr_lhs` (a Stan index type or
 * `matrix_cl<int>` instead of `index_multi`)
 * @throw std::out_of_range If the index is out of bounds.
 * @throw std::invalid_argument If the right hand side size isn't the same as
 * the indexed left hand side size.
 */
template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_nonscalar_prim_or_rev_kernel_expression_t<ExprRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ExprRhs& expr_rhs,
                   const char* name, RowIndex&& row_index) {
  decltype(auto) lhs_val = rvalue(expr_lhs.val_op(), name, row_index);
  stan::math::check_size_match(internal::print_index_type(row_index),
                               "left hand side rows", lhs_val.rows(), name,
                               expr_rhs.rows());
  stan::math::check_size_match(internal::print_index_type(row_index),
                               "left hand side columns", lhs_val.cols(), name,
                               expr_rhs.cols());
  math::arena_matrix_cl<double> prev_vals = lhs_val;
  lhs_val = math::value_of(expr_rhs);  // assign the values
  math::reverse_pass_callback(
      [expr_lhs, expr_rhs, name, prev_vals,
       row_index
       = math::to_arena(std::forward<RowIndex>(row_index))]() mutable {
        auto&& lhs_val = rvalue(expr_lhs.val_op(), name, row_index);
        decltype(auto) lhs_adj = rvalue(expr_lhs.adj(), name, row_index);

        math::results(lhs_val, math::adjoint_of(expr_rhs), lhs_adj)
            = math::expressions(
                prev_vals,
                math::calc_if<!is_constant<ExprRhs>::value>(
                    math::adjoint_of(expr_rhs) + lhs_adj),
                math::constant(0.0, lhs_adj.rows(), lhs_adj.cols()));
      });
}
// the "forwarding" overload
inline void assign(math::var_value<math::matrix_cl<double>>& expr_lhs,
                   math::var_value<math::matrix_cl<double>>&& expr_rhs,
                   const char* /*name*/, index_omni /*row_index*/) {
  expr_lhs.vi_ = expr_rhs.vi_;
}

/**
 * Assign one primitive or reverse mode kernel generator expression to a reverse
 * mode one, using given indices.
 *
 * @tparam ExprLhs type of the assignable rev expression on the left hand side
 * of the assignment
 * @tparam ExprRhs type of the prim or rev expression on the right hand side of
 * the assignment
 * @tparam RowIndex type of row index (a Stan index type or `matrix_cl<int>`
 * instead of `index_multi`)
 * @tparam ColIndex type of column index (a Stan index type or `matrix_cl<int>`
 * instead of `index_multi`)
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param expr_rhs expression on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing rows of `expr_lhs`
 * @param col_index index used for indexing columns of `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 * @throw std::invalid_argument If the right hand side size isn't the same as
 * the indexed left hand side size.
 */
template <
    typename ExprLhs, typename ExprRhs, typename RowIndex, typename ColIndex,
    require_rev_kernel_expression_t<ExprLhs>* = nullptr,
    require_all_nonscalar_prim_or_rev_kernel_expression_t<ExprRhs>* = nullptr,
    require_any_not_same_t<RowIndex, ColIndex, index_uni>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ExprRhs& expr_rhs,
                   const char* name, RowIndex&& row_index,
                   ColIndex&& col_index) {
  decltype(auto) lhs = rvalue(expr_lhs.val_op(), name, row_index, col_index);
  stan::math::check_size_match(internal::print_index_type(row_index),
                               "left hand side rows", lhs.rows(), name,
                               expr_rhs.rows());
  stan::math::check_size_match(internal::print_index_type(col_index),
                               "left hand side cols", lhs.cols(), name,
                               expr_rhs.cols());

  math::arena_matrix_cl<double> prev_vals = lhs;
  lhs = math::value_of(expr_rhs);  // assign the values
  math::reverse_pass_callback(
      [expr_lhs, expr_rhs, name, prev_vals,
       row_index = math::to_arena(std::forward<RowIndex>(row_index)),
       col_index
       = math::to_arena(std::forward<ColIndex>(col_index))]() mutable {
        decltype(auto) lhs_val
            = rvalue(expr_lhs.val_op(), name, row_index, col_index);
        decltype(auto) lhs_adj
            = rvalue(expr_lhs.adj(), name, row_index, col_index);
        math::results(lhs_val, math::adjoint_of(expr_rhs), lhs_adj)
            = math::expressions(
                prev_vals,
                math::calc_if<!is_constant<ExprRhs>::value>(
                    math::adjoint_of(expr_rhs) + lhs_adj),
                math::constant(0.0, lhs_adj.rows(), lhs_adj.cols()));
      });
}
// the "forwarding" overload
inline void assign(math::var_value<math::matrix_cl<double>>& expr_lhs,
                   math::var_value<math::matrix_cl<double>>&& expr_rhs,
                   const char* /*name*/, index_omni /*row_index*/,
                   index_omni /*col_index*/) {
  expr_lhs.vi_ = expr_rhs.vi_;
}

/**
 * Assign a primitive or reverse mode scalar to a reverse mode kernel generator
 * expression, using given uni indices.
 *
 * @tparam ExprLhs type of the assignable rev expression on the left hand side
 * of the assignment
 * @tparam ScalRhs type of the prim or rev scalar on the right hand side of the
 * assignment
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param scal_rhs scalar on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing rows of `expr_lhs`
 * @param col_index index used for indexing columns of `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename ExprLhs, typename ScalRhs,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_stan_scalar_t<ScalRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ScalRhs& scal_rhs,
                   const char* name, const index_uni row_index,
                   const index_uni col_index) {
  decltype(auto) lhs_val = math::block_zero_based(
      expr_lhs.val_op(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
  math::arena_matrix_cl<double> prev_val = lhs_val;
  lhs_val = math::constant(math::value_of(scal_rhs), 1, 1);  // assign the value
  math::reverse_pass_callback(
      [expr_lhs, scal_rhs, row_index, col_index, prev_val]() mutable {
        auto&& lhs_val = math::block_zero_based(
            expr_lhs.val_op(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
        decltype(auto) lhs_adj = math::block_zero_based(
            expr_lhs.adj(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
        if (!is_constant<ScalRhs>::value) {
          math::adjoint_of(scal_rhs) += math::from_matrix_cl<double>(lhs_adj);
        }
        math::results(lhs_adj, lhs_val)
            = math::expressions(math::constant(0.0, 1, 1), prev_val);
      });
}

}  // namespace model
}  // namespace stan

#endif
#endif
