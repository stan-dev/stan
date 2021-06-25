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

template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          require_kernel_expression_lhs_t<ExprLhs>* = nullptr,
          require_all_kernel_expressions_and_none_scalar_t<ExprRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ExprRhs& expr_rhs,
                   const char* name, const RowIndex& row_index) {
  if (std::is_same<RowIndex, index_omni>::value) {
    stan::math::check_size_match("omni assign", "left hand side rows",
                                 expr_lhs.rows(), name, expr_rhs.rows());
    stan::math::check_size_match("omni assign", "left hand side columns",
                                 expr_lhs.cols(), name, expr_rhs.cols());
  }
  rvalue(expr_lhs, name, row_index) = expr_rhs;
}

template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          typename ColIndex,
          require_kernel_expression_lhs_t<ExprLhs>* = nullptr,
          require_all_kernel_expressions_and_none_scalar_t<ExprRhs>* = nullptr,
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
  rvalue(expr_lhs, name, row_index, col_index) = expr_rhs;
}

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

template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_all_kernel_expressions_and_none_scalar_t<ExprRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ExprRhs& expr_rhs,
                   const char* name, const RowIndex& row_index) {
  if (std::is_same<RowIndex, index_omni>::value) {
    stan::math::check_size_match("omni assign", "left hand side rows",
                                 expr_lhs.rows(), name, expr_rhs.rows());
    stan::math::check_size_match("omni assign", "left hand side columns",
                                 expr_lhs.cols(), name, expr_rhs.cols());
  }
  auto&& lhs_holder = rvalue(expr_lhs.val_op(), name, row_index);
  auto& lhs = lhs_holder;
  math::arena_matrix_cl<double> prev_vals = lhs;
  lhs = expr_rhs;
  math::reverse_pass_callback([expr_lhs, row_index, prev_vals, name]() mutable {
    auto&& lhs_val = rvalue(expr_lhs.val_op(), name, row_index);
    auto&& lhs_adj = rvalue(expr_lhs.adj(), name, row_index);
    math::results(lhs_val, lhs_adj) = math::expressions(
        prev_vals, math::constant(0.0, lhs_adj.rows(), lhs_adj.cols()));
  });
}

template <typename ExprLhs, typename ExprRhs, typename RowIndex,
          typename ColIndex,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_all_kernel_expressions_and_none_scalar_t<ExprRhs>* = nullptr,
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
  auto&& lhs_holder = rvalue(expr_lhs.val_op(), name, row_index, col_index);
  auto& lhs = lhs_holder;
  math::arena_matrix_cl<double> prev_vals = lhs;
  lhs = expr_rhs;
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

template <typename ExprLhs, typename ScalRhs,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_arithmetic_t<ScalRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ScalRhs& scal_rhs,
                   const char* name, const index_uni row_index,
                   const index_uni col_index) {
  auto&& lhs_holder = math::block_zero_based(
      expr_lhs.val_op(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
  auto& lhs = lhs_holder;
  math::arena_matrix_cl<double> prev_val = lhs;
  lhs = math::constant(scal_rhs, 1, 1);
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
  auto&& lhs_holder = rvalue(expr_lhs.val_op(), name, row_index);
  auto& lhs = lhs_holder;
  math::arena_matrix_cl<double> prev_vals = lhs;
  lhs = expr_rhs.val();
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
  auto&& lhs_holder = rvalue(expr_lhs.val_op(), name, row_index, col_index);
  auto& lhs = lhs_holder;
  math::arena_matrix_cl<double> prev_vals = lhs;
  lhs = expr_rhs.val();
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

template <typename ExprLhs, typename ScalRhs,
          require_rev_kernel_expression_t<ExprLhs>* = nullptr,
          require_var_t<ScalRhs>* = nullptr>
inline void assign(ExprLhs&& expr_lhs, const ScalRhs& scal_rhs,
                   const char* name, const index_uni row_index,
                   const index_uni col_index) {
  auto&& lhs_holder = math::block_zero_based(
      expr_lhs.val_op(), row_index.n_ - 1, col_index.n_ - 1, 1, 1);
  auto& lhs = lhs_holder;
  math::arena_matrix_cl<double> prev_val = lhs;
  lhs = math::constant(scal_rhs.val(), 1, 1);
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
