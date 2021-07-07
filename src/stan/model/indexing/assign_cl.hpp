#ifndef STAN_MODEL_INDEXING_ASSIGN_CL_HPP
#define STAN_MODEL_INDEXING_ASSIGN_CL_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/rev.hpp>
#include <stan/math/opencl/indexing_rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/rvalue_cl.hpp>
#include <tuple>
#include <utility>

namespace stan {
namespace model {

// prim
/**
 * Assign one primitive kernel generator expression to another, using given
 * index.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ExprRhs type of the expression on the right hand side of the
 * assignment
 * @tparam RowIndex type of index
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
  if (std::is_same<RowIndex, index_omni>::value) {
    stan::math::check_size_match("omni assign", "left hand side rows",
                                 expr_lhs.rows(), name, expr_rhs.rows());
    stan::math::check_size_match("omni assign", "left hand side columns",
                                 expr_lhs.cols(), name, expr_rhs.cols());
  }
  rvalue(expr_lhs, name, row_index) = std::forward<ExprRhs>(expr_rhs);
}

/**
 * Assign one primitive kernel generator expression to another, using given
 * indices.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ExprRhs type of the expression on the right hand side of the
 * assignment
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of columnindex
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
  if (std::is_same<RowIndex, index_omni>::value
      && std::is_same<ColIndex, index_omni>::value) {
    stan::math::check_size_match("omni assign", "left hand side rows",
                                 expr_lhs.rows(), name, expr_rhs.rows());
    stan::math::check_size_match("omni assign", "left hand side columns",
                                 expr_lhs.cols(), name, expr_rhs.cols());
  }
  rvalue(expr_lhs, name, row_index, col_index)
      = std::forward<ExprRhs>(expr_rhs);
}

/**
 * Assign a scalar to a primitive kernel generator expression, using given uni
 * indices.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ScalRhs type of the scalar on the right hand side of the assignment
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

// assign(rev, prim, ...)
/**
 * Assign one primitive kernel generator expression to a reverse mode kernel
 * generator expression, using given index.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ExprRhs type of the expression on the right hand side of the
 * assignment
 * @tparam RowIndex type of index
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param expr_rhs expression on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 * @throw std::invalid_argument If the right hand side size isn't the same as
 * the indexed left hand side size.
 */
template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_all_kernel_expressions_and_none_scalar_t<ExprRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, ExprRhs&& expr_rhs, const char* name,
                   const RowIndex& row_index) {
  if (std::is_same<RowIndex, index_omni>::value) {
    stan::math::check_size_match("omni assign", "left hand side rows",
                                 expr_lhs.rows(), name, expr_rhs.rows());
    stan::math::check_size_match("omni assign", "left hand side columns",
                                 expr_lhs.cols(), name, expr_rhs.cols());
  }
  decltype(auto) lhs = rvalue(expr_lhs.val_op(), name, row_index);
  math::arena_matrix_cl<double> prev_vals;
  if (std::is_rvalue_reference<ExprRhs&&>::value) {
    prev_vals = std::move(lhs);
  } else {
    prev_vals = lhs;
  }
  lhs = std::forward<ExprRhs>(expr_rhs); //assign the values
  math::reverse_pass_callback([expr_lhs, row_index, prev_vals, name]() mutable {
    auto&& lhs_val = rvalue(expr_lhs.val_op(), name, row_index);
    auto&& lhs_adj = rvalue(expr_lhs.adj(), name, row_index);
    math::results(lhs_val, lhs_adj) = math::expressions(
        prev_vals, math::constant(0.0, lhs_adj.rows(), lhs_adj.cols()));
  });
}

/**
 * Assign one primitive kernel generator expression to a reverse mode kernel
 * generator expression, using given indices.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ExprRhs type of the expression on the right hand side of the
 * assignment
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of columnindex
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
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_all_kernel_expressions_and_none_scalar_t<ExprRhs>* = nullptr,
          require_any_not_same_t<RowIndex, ColIndex, index_uni>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, ExprRhs&& expr_rhs, const char* name,
                   const RowIndex& row_index, const ColIndex& col_index) {
  if (std::is_same<RowIndex, index_omni>::value
      && std::is_same<ColIndex, index_omni>::value) {
    stan::math::check_size_match("omni assign", "left hand side rows",
                                 expr_lhs.rows(), name, expr_rhs.rows());
    stan::math::check_size_match("omni assign", "left hand side columns",
                                 expr_lhs.cols(), name, expr_rhs.cols());
  }
  decltype(auto) lhs = rvalue(expr_lhs.val_op(), name, row_index, col_index);
  math::arena_matrix_cl<double> prev_vals;
  if (std::is_rvalue_reference<ExprRhs&&>::value) {
    prev_vals = std::move(lhs);
  } else {
    prev_vals = lhs;
  }
  lhs = std::forward<ExprRhs>(expr_rhs); //assign the values
  math::reverse_pass_callback(
      [expr_lhs, row_index, col_index, prev_vals, name]() mutable {
        auto&& lhs_val = rvalue(expr_lhs.val_op(), name, row_index, col_index);
        auto&& lhs_adj = rvalue(expr_lhs.adj(), name, row_index, col_index);
        lhs_adj = math::constant(0.0, lhs_adj.rows(), lhs_adj.cols());
        lhs_val = prev_vals;
        math::results(lhs_val, lhs_adj) = math::expressions(
            prev_vals, math::constant(0.0, lhs_adj.rows(), lhs_adj.cols()));
      });
}

/**
 * Assign a primitive scalar to a reverse mode kernel generator expression,
 * using given uni indices.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ScalRhs type of the scalar on the right hand side of the assignment
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param scal_rhs scalar on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing rows of `expr_lhs`
 * @param col_index index used for indexing columns of `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename ExprLhs, typename ScalRhs,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_arithmetic_t<ScalRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ScalRhs& scal_rhs,
                   const char* name, const index_uni row_index,
                   const index_uni col_index) {
  decltype(auto) lhs = math::block_zero_based(
      expr_lhs.val_op(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
  math::arena_matrix_cl<double> prev_val = lhs;
  lhs = math::constant(scal_rhs, 1, 1); //assign the values
  math::reverse_pass_callback(
      [expr_lhs, row_index, col_index, prev_val]() mutable {
        auto&& lhs_val = math::block_zero_based(
            expr_lhs.val_op(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
        auto&& lhs_adj = math::block_zero_based(
            expr_lhs.adj(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
        math::results(lhs_adj, lhs_val)
            = math::expressions(math::constant(0.0, 1, 1), prev_val);
      });
}

// assign(rev, rev, ...)
/**
 * Assign one reverse mode kernel generator expression to another, using given
 * index.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ExprRhs type of the expression on the right hand side of the
 * assignment
 * @tparam RowIndex type of index
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param expr_rhs expression on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 * @throw std::invalid_argument If the right hand side size isn't the same as
 * the indexed left hand side size.
 */
template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          require_all_rev_kernel_expression_t<ExprLhs, ExprRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ExprRhs& expr_rhs,
                   const char* name, const RowIndex& row_index) {
  if (std::is_same<RowIndex, index_omni>::value) {
    stan::math::check_size_match("omni assign", "left hand side rows",
                                 expr_lhs.rows(), name, expr_rhs.rows());
    stan::math::check_size_match("omni assign", "left hand side columns",
                                 expr_lhs.cols(), name, expr_rhs.cols());
  }
  decltype(auto) lhs = rvalue(expr_lhs.val_op(), name, row_index);
  math::arena_matrix_cl<double> prev_vals = lhs;
  lhs = expr_rhs.val(); //assign the values
  math::reverse_pass_callback(
      [expr_lhs, expr_rhs, row_index, name, prev_vals]() mutable {
        auto&& lhs_val = rvalue(expr_lhs.val_op(), name, row_index);
        auto&& lhs_adj_holder = rvalue(expr_lhs.adj(), name, row_index);
        auto& lhs_adj = lhs_adj_holder;

        math::results(lhs_val, expr_rhs.adj(), lhs_adj) = math::expressions(
            prev_vals, expr_rhs.adj() + lhs_adj,
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
 * Assign one reverse mode kernel generator expression to another, using given
 * indices.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ExprRhs type of the expression on the right hand side of the
 * assignment
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of columnindex
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
          require_all_rev_kernel_expression_t<ExprLhs, ExprRhs>* = nullptr,
          require_any_not_same_t<RowIndex, ColIndex, index_uni>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ExprRhs& expr_rhs,
                   const char* name, const RowIndex& row_index,
                   const ColIndex& col_index) {
  if (std::is_same<RowIndex, index_omni>::value
      && std::is_same<ColIndex, index_omni>::value) {
    stan::math::check_size_match("omni assign", "left hand side rows",
                                 expr_lhs.rows(), name, expr_rhs.rows());
    stan::math::check_size_match("omni assign", "left hand side columns",
                                 expr_lhs.cols(), name, expr_rhs.cols());
  }
  decltype(auto) lhs = rvalue(expr_lhs.val_op(), name, row_index, col_index);
  math::arena_matrix_cl<double> prev_vals = lhs;
  lhs = expr_rhs.val(); //assign the values
  math::reverse_pass_callback([expr_lhs, expr_rhs, row_index, col_index, name,
                               prev_vals]() mutable {
    auto&& lhs_val_holder
        = rvalue(expr_lhs.val_op(), name, row_index, col_index);
    auto&& lhs_adj_holder = rvalue(expr_lhs.adj(), name, row_index, col_index);
    auto& lhs_val = lhs_val_holder;
    auto& lhs_adj = lhs_adj_holder;
    math::results(lhs_val, expr_rhs.adj(), lhs_adj) = math::expressions(
        prev_vals, expr_rhs.adj() + lhs_adj,
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
 * Assign a reverse mode scalar to a reverse mode kernel generator expression,
 * using given uni indices.
 *
 * @tparam ExprLhs type of the expression on the left hand side of the
 * assignment
 * @tparam ScalRhs type of the scalar on the right hand side of the assignment
 * @param[in,out] expr_lhs expression on the left hand side of the assignment
 * @param scal_rhs scalar on the right hand side of the assignment
 * @param name Name of lvalue variable
 * @param row_index index used for indexing rows of `expr_lhs`
 * @param col_index index used for indexing columns of `expr_lhs`
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename ExprLhs, typename ScalRhs,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_var_t<ScalRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ScalRhs& scal_rhs,
                   const char* name, const index_uni row_index,
                   const index_uni col_index) {
  decltype(auto) lhs = math::block_zero_based(
      expr_lhs.val_op(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
  math::arena_matrix_cl<double> prev_val = lhs;
  lhs = math::constant(scal_rhs.val(), 1, 1); //assign the values
  math::reverse_pass_callback(
      [expr_lhs, scal_rhs, row_index, col_index, prev_val]() mutable {
        auto&& lhs_val = math::block_zero_based(
            expr_lhs.val_op(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
        auto&& lhs_adj = math::block_zero_based(
            expr_lhs.adj(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
        scal_rhs.adj() += math::from_matrix_cl<double>(lhs_adj);
        math::results(lhs_adj, lhs_val)
            = math::expressions(math::constant(0.0, 1, 1), prev_val);
      });
}

}  // namespace model
}  // namespace stan

#endif
#endif
