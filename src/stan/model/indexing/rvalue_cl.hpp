#ifndef STAN_MODEL_INDEXING_RVALUE_CL_HPP
#define STAN_MODEL_INDEXING_RVALUE_CL_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/rev.hpp>
#include <stan/math/opencl/indexing_rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <tuple>
#include <utility>

namespace stan {
namespace model {
namespace internal {

inline auto cl_row_index(index_uni i, int rows, const char* name) {
  math::check_range("uni indexing", name, rows, i.n_);
  return math::constant(i.n_ - 1, 1, -1);
}
inline auto cl_row_index(const math::matrix_cl<int>& i, int rows,
                         const char* name) {
  return math::rowwise_broadcast(i - 1);
}
inline auto cl_row_index(index_omni i, int rows, const char* name) {
  return math::row_index(rows, -1);
}
inline auto cl_row_index(index_min i, int rows, const char* name) {
  stan::math::check_range("min indexing", name, rows, i.min_);
  return math::row_index(rows - (i.min_ - 1), -1) + (i.min_ - 1);
}
inline auto cl_row_index(index_max i, int rows, const char* name) {
  stan::math::check_range("max indexing", name, rows, i.max_);
  return math::row_index(i.max_, -1);
}
inline auto cl_row_index(index_min_max i, int rows, const char* name) {
  math::check_range("min_max indexing min", name, rows, i.min_);
  math::check_range("min_max indexing max", name, rows, i.max_);
  if (i.min_ <= i.max_) {
    return 1 * math::row_index(i.max_ - (i.min_ - 1), -1) + (i.min_ - 1);
  } else {
    return -1 * math::row_index(i.min_ - (i.max_ - 1), -1) + (i.min_ - 1);
  }
}

inline auto cl_col_index(index_uni i, int cols, const char* name) {
  math::check_range("uni indexing", name, cols, i.n_);
  return math::constant(i.n_ - 1, -1, 1);
}
inline auto cl_col_index(const math::matrix_cl<int>& i, int cols,
                         const char* name) {
  return math::colwise_broadcast(math::transpose(i - 1));
}
inline auto cl_col_index(index_omni i, int cols, const char* name) {
  return math::col_index(-1, cols);
}
inline auto cl_col_index(index_min i, int cols, const char* name) {
  stan::math::check_range("min indexing", name, cols, i.min_);
  return math::col_index(-1, cols - (i.min_ - 1)) + (i.min_ - 1);
}
inline auto cl_col_index(index_max i, int cols, const char* name) {
  stan::math::check_range("max indexing", name, cols, i.max_);
  return math::col_index(-1, i.max_);
}
inline auto cl_col_index(index_min_max i, int cols, const char* name) {
  math::check_range("min_max indexing min", name, cols, i.min_);
  math::check_range("min_max indexing max", name, cols, i.max_);
  if (i.min_ <= i.max_) {
    return 1 * math::col_index(-1, i.max_ - (i.min_ - 1)) + (i.min_ - 1);
  } else {
    return -1 * math::col_index(-1, i.min_ - (i.max_ - 1)) + (i.min_ - 1);
  }
}

template <typename Index>
inline std::tuple<> index_check(Index i, const char* name, int dim) {
  return std::make_tuple();
}
inline auto index_check(const math::matrix_cl<int>& i, const char* name,
                        int dim) {
  return std::make_tuple(std::make_pair(
      math::check_cl("multi-indexing", name, i, "within range"),
      static_cast<int>(stan::error_index::value) <= i
          && i < dim + static_cast<int>(stan::error_index::value)));
}
}  // namespace internal

// prim
/**
 * Index a prim kernel generator expression with one index.
 *
 * @tparam Expr type of the exoression
 * @tparam RowIndex type of index
 * @param expr a prim kernel generator expression to index
 * @param name string form of expression being evaluated
 * @param row_index index
 * @return result of indexing
 */
template <typename Expr, typename RowIndex,
          require_all_kernel_expressions_and_none_scalar_t<Expr>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const RowIndex& row_index) {
  auto checks = internal::index_check(row_index, name, expr.rows());
  try {
    math::index_apply<std::tuple_size<decltype(checks)>::value>(
        [&checks](auto... Is) {
          math::results(std::get<Is>(checks).first...)
              = math::expressions(std::get<Is>(checks).second...);
        });
  } catch (const std::domain_error& e) {
    throw std::out_of_range(e.what());
  }
  return math::indexing(expr,
                        internal::cl_row_index(row_index, expr.rows(), name),
                        math::col_index(-1, expr.cols()));
}

/**
 * Index a prim kernel generator expression with two indices.
 *
 * @tparam Expr type of the exoression
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of column index
 * @param expr a prim kernel generator expression to index
 * @param name string form of expression being evaluated
 * @param row_index row index
 * @param col_index column index
 * @return result of indexing
 */
template <typename Expr, typename RowIndex, typename ColIndex,
          require_all_kernel_expressions_and_none_scalar_t<Expr>* = nullptr,
          require_any_not_same_t<RowIndex, ColIndex, index_uni>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const RowIndex& row_index,
                   const ColIndex& col_index) {
  auto checks1 = internal::index_check(row_index, name, expr.rows());
  auto checks2 = internal::index_check(col_index, name, expr.cols());
  try {
    math::index_apply<std::tuple_size<decltype(checks1)>::value>(
        [&checks1](auto... Is) {
          math::results(std::get<Is>(checks1).first...)
              = math::expressions(std::get<Is>(checks1).second...);
        });
    math::index_apply<std::tuple_size<decltype(checks2)>::value>(
        [&checks2](auto... Is) {
          math::results(std::get<Is>(checks2).first...)
              = math::expressions(std::get<Is>(checks2).second...);
        });
  } catch (const std::domain_error& e) {
    throw std::out_of_range(e.what());
  }

  return math::indexing(expr,
                        internal::cl_row_index(row_index, expr.rows(), name),
                        internal::cl_col_index(col_index, expr.cols(), name));
}

/**
 * Index a prim kernel generator expression with two single indices.
 *
 * @tparam Expr type of the exoression
 * @param expr a prim kernel generator expression to index
 * @param name string form of expression being evaluated
 * @param row_index row index
 * @param col_index column index
 * @return result of indexing (scalar)
 */
template <typename Expr,
          require_all_kernel_expressions_and_none_scalar_t<Expr>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const index_uni row_index,
                   const index_uni col_index) {
  using Val = stan::value_type_t<Expr>;
  math::matrix_cl<Val> expr_eval = expr;
  math::check_range("uni indexing", name, expr_eval.rows(), row_index.n_);
  math::check_range("uni indexing", name, expr_eval.cols(), col_index.n_);
  cl::CommandQueue queue = stan::math::opencl_context.queue();
  Val res;
  try {
    cl::Event copy_event;
    queue.enqueueReadBuffer(
        expr_eval.buffer(), true,
        sizeof(Val)
            * (row_index.n_ - 1 + (col_index.n_ - 1) * expr_eval.rows()),
        sizeof(Val), &res, &expr_eval.write_events(), &copy_event);
    copy_event.wait();
  } catch (const cl::Error& e) {
    stan::math::check_opencl_error("uni uni indexing", e);
  }
  return res;
}

// rev, without multi-index - no data races
/**
 * Index a rev kernel generator expression with one (non multi-) index.
 *
 * @tparam Expr type of the exoression
 * @tparam RowIndex type of index
 * @param expr a prim kernel generator expression to index
 * @param name string form of expression being evaluated
 * @param row_index index
 * @return result of indexing
 */
template <typename Expr, typename RowIndex,
          require_rev_kernel_expression_t<Expr>* = nullptr,
          require_not_same_t<RowIndex, math::matrix_cl<int>>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const RowIndex row_index) {
  int rows = expr.rows();
  auto checks1 = internal::index_check(row_index, name, rows);
  try {
    math::index_apply<std::tuple_size<decltype(checks1)>::value>(
        [&checks1](auto... Is) {
          math::results(std::get<Is>(checks1).first...)
              = math::expressions(std::get<Is>(checks1).second...);
        });
  } catch (const std::domain_error& e) {
    throw std::out_of_range(e.what());
  }
  auto res_vari
      = expr.vi_->index(internal::cl_row_index(row_index, expr.rows(), name),
                        math::col_index(-1, expr.cols()));
  return math::var_value<value_type_t<decltype(res_vari)>>(
      new decltype(res_vari)(res_vari));
}

/**
 * Index a rev kernel generator expression with two (non-multi) indices.
 *
 * @tparam Expr type of the exoression
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of column index
 * @param expr a prim kernel generator expression to index
 * @param name string form of expression being evaluated
 * @param row_index row index
 * @param col_index column index
 * @return result of indexing
 */
template <typename Expr, typename RowIndex, typename ColIndex,
          require_rev_kernel_expression_t<Expr>* = nullptr,
          require_not_same_t<RowIndex, math::matrix_cl<int>>* = nullptr,
          require_not_same_t<ColIndex, math::matrix_cl<int>>* = nullptr,
          require_any_not_same_t<RowIndex, ColIndex, index_uni>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const RowIndex row_index,
                   const ColIndex col_index) {
  int rows = expr.rows();
  int cols = expr.cols();
  auto checks1 = internal::index_check(row_index, name, rows);
  auto checks2 = internal::index_check(col_index, name, cols);
  try {
    math::index_apply<std::tuple_size<decltype(checks1)>::value>(
        [&checks1](auto... Is) {
          math::results(std::get<Is>(checks1).first...)
              = math::expressions(std::get<Is>(checks1).second...);
        });
    math::index_apply<std::tuple_size<decltype(checks2)>::value>(
        [&checks2](auto... Is) {
          math::results(std::get<Is>(checks2).first...)
              = math::expressions(std::get<Is>(checks2).second...);
        });
  } catch (const std::domain_error& e) {
    throw std::out_of_range(e.what());
  }

  auto res_vari
      = expr.vi_->index(internal::cl_row_index(row_index, rows, name),
                        internal::cl_col_index(col_index, expr.cols(), name));
  return math::var_value<value_type_t<decltype(res_vari)>>(
      new decltype(res_vari)(res_vari));
}

/**
 * Index a rev kernel generator expression with two uni indices.
 *
 * @tparam Expr type of the exoression
 * @param expr a prim kernel generator expression to index
 * @param name string form of expression being evaluated
 * @param row_index row index
 * @param col_index column index
 * @return result of indexing
 */
template <typename Expr, require_rev_kernel_expression_t<Expr>* = nullptr>
inline math::var rvalue(Expr&& expr, const char* name,
                        const index_uni row_index, const index_uni col_index) {
  using Val = stan::value_type_t<stan::value_type_t<Expr>>;
  math::check_range("uni indexing", name, expr.rows(), row_index.n_);
  math::check_range("uni indexing", name, expr.cols(), col_index.n_);
  cl::CommandQueue queue = stan::math::opencl_context.queue();
  Val res;
  try {
    cl::Event copy_event;
    queue.enqueueReadBuffer(
        expr.val().buffer(), true,
        sizeof(Val) * (row_index.n_ - 1 + (col_index.n_ - 1) * expr.rows()),
        sizeof(Val), &res, &expr.val().write_events(), &copy_event);
    copy_event.wait();
  } catch (const cl::Error& e) {
    stan::math::check_opencl_error("uni uni indexing", e);
  }
  return math::make_callback_var(
      res, [expr, row_index, col_index](math::vari res_vari) mutable {
        block(expr.adj(), row_index.n_, col_index.n_, 1, 1)
            += math::constant(res_vari.adj(), 1, 1);
      });
}

// rev, with multi-index - possible data races in rev, needs special kernel
/**
 * Index a rev kernel generator expression with one multi-index.
 *
 * @tparam Expr type of the exoression
 * @tparam RowIndex type of index
 * @param expr a prim kernel generator expression to index
 * @param name string form of expression being evaluated
 * @param row_index index
 * @return result of indexing
 */
template <typename Expr, require_rev_kernel_expression_t<Expr>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name,
                   const math::matrix_cl<int>& row_index) {
  auto row_idx_expr = math::rowwise_broadcast(row_index - 1);
  auto col_idx_expr = math::col_index(-1, expr.cols());
  auto res_expr = math::indexing(expr.val(), row_idx_expr, col_idx_expr);
  auto lin_idx_expr
      = row_idx_expr + col_idx_expr * static_cast<int>(expr.rows());

  try {
    math::check_cl("multi-indexing", name, row_index, "within range")
        = static_cast<int>(stan::error_index::value) <= row_index
          && row_index
                 < static_cast<int>(expr.rows() + stan::error_index::value);
  } catch (const std::domain_error& e) {
    throw std::out_of_range(e.what());
  }

  math::matrix_cl<double> res;
  math::arena_matrix_cl<int> lin_idx;
  math::results(res, lin_idx) = math::expressions(res_expr, lin_idx_expr);
  return make_callback_var(
      res, [expr, lin_idx](
               math::vari_value<math::matrix_cl<double>>& res_vari) mutable {
        math::indexing_rev(expr.adj(), lin_idx, res_vari.adj());
      });
}

/**
 * Index a rev kernel generator expression with two indices, at least one of
 * which is multi-index.
 *
 * @tparam Expr type of the exoression
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of column index
 * @param expr a prim kernel generator expression to index
 * @param name string form of expression being evaluated
 * @param row_index row index
 * @param col_index column index
 * @return result of indexing
 */
template <
    typename Expr, typename RowIndex, typename ColIndex,
    require_rev_kernel_expression_t<Expr>* = nullptr,
    require_any_t<std::is_same<RowIndex, math::matrix_cl<int>>,
                  std::is_same<ColIndex, math::matrix_cl<int>>>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const RowIndex& row_index,
                   const ColIndex& col_index) {
  int rows = expr.rows();
  int cols = expr.cols();
  auto checks1 = internal::index_check(row_index, name, rows);
  auto checks2 = internal::index_check(col_index, name, cols);
  try {
    math::index_apply<std::tuple_size<decltype(checks1)>::value>(
        [&checks1](auto... Is) {
          math::results(std::get<Is>(checks1).first...)
              = math::expressions(std::get<Is>(checks1).second...);
        });
    math::index_apply<std::tuple_size<decltype(checks2)>::value>(
        [&checks2](auto... Is) {
          math::results(std::get<Is>(checks2).first...)
              = math::expressions(std::get<Is>(checks2).second...);
        });
  } catch (const std::domain_error& e) {
    throw std::out_of_range(e.what());
  }
  auto row_idx_expr = internal::cl_row_index(row_index, rows, name);
  auto col_idx_expr = internal::cl_col_index(col_index, cols, name);
  auto res_expr = math::indexing(expr.val(), row_idx_expr, col_idx_expr);
  auto lin_idx_expr = row_idx_expr + col_idx_expr * rows;
  math::matrix_cl<double> res;
  math::arena_matrix_cl<int> lin_idx;
  math::results(res, lin_idx) = math::expressions(res_expr, lin_idx_expr);
  return make_callback_var(
      res, [expr, lin_idx](
               math::vari_value<math::matrix_cl<double>>& res_vari) mutable {
        math::indexing_rev(expr.adj(), lin_idx, res_vari.adj());
      });
}

}  // namespace model
}  // namespace stan
#endif
#endif
