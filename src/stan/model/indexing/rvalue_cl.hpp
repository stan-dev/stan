#ifndef STAN_MODEL_INDEXING_RVALUE_CL_HPP
#define STAN_MODEL_INDEXING_RVALUE_CL_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/rev.hpp>
#include <stan/math/opencl/indexing_rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <utility>

namespace stan {
namespace model {
namespace internal {

inline auto cl_row_index(index_uni i, int rows, const char* name) {
  math::check_range("uni indexing", name, rows, i.n_);
  return math::constant(i.n_ - 1, 1, -1);
}
template <typename T, require_matrix_cl_t<T>* = nullptr,
          require_all_vt_same<T, int>* = nullptr>
inline auto cl_row_index(const T& i, int rows, const char* name) {
  return math::rowwise_broadcast(i - 1);
}
inline auto cl_row_index(index_omni /*i*/, int rows, const char* name) {
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
    return 1 * math::row_index(0, -1) + (i.min_ - 1);
  }
}

inline auto cl_col_index(index_uni i, int cols, const char* name) {
  math::check_range("uni indexing", name, cols, i.n_);
  return math::constant(i.n_ - 1, -1, 1);
}
template <typename T, require_matrix_cl_t<T>* = nullptr,
          require_all_vt_same<T, int>* = nullptr>
inline auto cl_col_index(const T& i, int cols, const char* name) {
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
    return 1 * math::col_index(-1, 0) + (i.min_ - 1);
  }
}

template <typename Index>
inline void index_check(Index i, const char* name, int dim) {}

inline void index_check(const index_min_max& i, const char* name, int dim) {
  math::check_range("min_max indexing min", name, dim, i.min_);
  math::check_range("min_max indexing max", name, dim, i.max_);
}

inline void index_check(const index_max& i, const char* name, int dim) {
  math::check_range("min_max indexing min", name, dim, i.max_);
}
inline void index_check(const index_min& i, const char* name, int dim) {
  math::check_range("min_max indexing", name, dim, i.min_);
}
inline void index_check(const index_uni& i, const char* name, int dim) {
  math::check_range("uni indexing", name, dim, i.n_);
}

inline void index_check(const math::matrix_cl<int>& i, const char* name,
                        int dim) {
  try {
    math::check_cl("multi-indexing", name, i, "within range")
        = static_cast<int>(stan::error_index::value) <= i
          && i < dim + static_cast<int>(stan::error_index::value);
  } catch (const std::domain_error& e) {
    throw std::out_of_range(e.what());
  }
}
}  // namespace internal

// prim
/**
 * Index a prim kernel generator expression with one index.
 *
 * @tparam Expr type of the expression
 * @tparam RowIndex type of index
 * @param expr a prim kernel generator expression to index
 * @param name name of value being indexed (if named, otherwise an empty string)
 * @param row_index index
 * @return result of indexing
 */
template <typename Expr, typename RowIndex,
          require_all_kernel_expressions_and_none_scalar_t<Expr>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const RowIndex& row_index) {
  internal::index_check(row_index, name, expr.rows());
  return math::indexing(expr,
                        internal::cl_row_index(row_index, expr.rows(), name),
                        math::col_index(-1, expr.cols()));
}

/**
 * Index a prim kernel generator expression with two indices.
 *
 * @tparam Expr type of the expression
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of column index
 * @param expr a prim kernel generator expression to index
 * @param name name of value being indexed (if named, otherwise an empty string)
 * @param row_index row index
 * @param col_index column index
 * @return result of indexing
 */
template <typename Expr, typename RowIndex, typename ColIndex,
          require_all_kernel_expressions_and_none_scalar_t<Expr>* = nullptr,
          require_any_not_same_t<RowIndex, ColIndex, index_uni>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const RowIndex& row_index,
                   const ColIndex& col_index) {
  internal::index_check(row_index, name, expr.rows());
  internal::index_check(col_index, name, expr.cols());
  return math::indexing(expr,
                        internal::cl_row_index(row_index, expr.rows(), name),
                        internal::cl_col_index(col_index, expr.cols(), name));
}

/**
 * Index a prim kernel generator expression with two single indices.
 *
 * @tparam Expr type of the expression
 * @param expr a prim kernel generator expression to index
 * @param name name of value being indexed (if named, otherwise an empty string)
 * @param row_index row index
 * @param col_index column index
 * @return result of indexing (scalar)
 */
template <typename Expr,
          require_all_kernel_expressions_and_none_scalar_t<Expr>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const index_uni row_index,
                   const index_uni col_index) {
  using Val = stan::value_type_t<Expr>;
  decltype(auto) expr_eval = expr.eval();
  math::check_range("uni indexing", name, expr_eval.rows(), row_index.n_);
  math::check_range("uni indexing", name, expr_eval.cols(), col_index.n_);
  Val res;
  try {
    cl::Event copy_event;
    std::vector<cl::Event> copy_write_events(expr_eval.write_events().begin(),
                                             expr_eval.write_events().end());
    cl::CommandQueue& queue = stan::math::opencl_context.queue();
    queue.enqueueReadBuffer(
        expr_eval.buffer(), true,
        sizeof(Val)
            * (row_index.n_ - 1 + (col_index.n_ - 1) * expr_eval.rows()),
        sizeof(Val), &res, &copy_write_events, &copy_event);
    copy_event.wait();
  } catch (const cl::Error& e) {
    std::ostringstream m;
    m << "uni uni indexing of " << name;
    stan::math::check_opencl_error(m.str().c_str(), e);
  }
  return res;
}

// rev, without multi-index - no data races
/**
 * Index a rev kernel generator expression with one (non multi-) index.
 *
 * @tparam Expr type of the expression
 * @tparam RowIndex type of index
 * @param expr a prim kernel generator expression to index
 * @param name name of value being indexed (if named, otherwise an empty string)
 * @param row_index index
 * @return result of indexing
 */
template <typename Expr, typename RowIndex,
          require_rev_kernel_expression_t<Expr>* = nullptr,
          require_not_same_t<RowIndex, math::matrix_cl<int>>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name, const RowIndex row_index) {
  int rows = expr.rows();
  internal::index_check(row_index, name, rows);
  auto res_vari
      = expr.vi_->index(internal::cl_row_index(row_index, expr.rows(), name),
                        math::col_index(-1, expr.cols()));
  return math::var_value<value_type_t<decltype(res_vari)>>(
      new decltype(res_vari)(std::move(res_vari)));
}

/**
 * Index a rev kernel generator expression with two (non-multi) indices.
 *
 * @tparam Expr type of the expression
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of column index
 * @param expr a prim kernel generator expression to index
 * @param name name of value being indexed (if named, otherwise an empty string)
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
  internal::index_check(row_index, name, rows);
  internal::index_check(col_index, name, cols);
  auto res_vari
      = expr.vi_->index(internal::cl_row_index(row_index, rows, name),
                        internal::cl_col_index(col_index, expr.cols(), name));
  return math::var_value<value_type_t<decltype(res_vari)>>(
      new decltype(res_vari)(std::move(res_vari)));
}

/**
 * Index a rev kernel generator expression with two uni indices.
 *
 * @tparam Expr type of the expression
 * @param expr a prim kernel generator expression to index
 * @param name name of value being indexed (if named, otherwise an empty string)
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
  Val res;
  try {
    std::vector<cl::Event> copy_write_events(expr.val().write_events().begin(),
                                             expr.val().write_events().end());
    cl::Event copy_event;
    cl::CommandQueue& queue = stan::math::opencl_context.queue();
    queue.enqueueReadBuffer(
        expr.val().buffer(), true,
        sizeof(Val) * (row_index.n_ - 1 + (col_index.n_ - 1) * expr.rows()),
        sizeof(Val), &res, &copy_write_events, &copy_event);
    copy_event.wait();
  } catch (const cl::Error& e) {
    std::ostringstream m;
    m << "uni uni indexing of " << name;
    stan::math::check_opencl_error(m.str().c_str(), e);
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
 * @tparam Expr type of the expression
 * @tparam RowIndex type of index
 * @param expr a prim kernel generator expression to index
 * @param name name of value being indexed (if named, otherwise an empty string)
 * @param row_index index
 * @return result of indexing
 */
template <typename Expr, require_rev_kernel_expression_t<Expr>* = nullptr>
inline auto rvalue(Expr&& expr, const char* name,
                   const math::matrix_cl<int>& row_index) {
  internal::index_check(row_index, name, expr.rows());
  auto row_idx_expr = math::rowwise_broadcast(row_index - 1);
  auto col_idx_expr = math::col_index(-1, expr.cols());
  auto res_expr = math::indexing(expr.val_op(), row_idx_expr, col_idx_expr);
  auto lin_idx_expr
      = row_idx_expr + col_idx_expr * static_cast<int>(expr.rows());

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
 * @tparam Expr type of the expression
 * @tparam RowIndex type of row index
 * @tparam ColIndex type of column index
 * @param expr a prim kernel generator expression to index
 * @param name name of value being indexed (if named, otherwise an empty string)
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
  internal::index_check(row_index, name, rows);
  internal::index_check(col_index, name, cols);
  auto row_idx_expr = internal::cl_row_index(row_index, rows, name);
  auto col_idx_expr = internal::cl_col_index(col_index, cols, name);
  auto res_expr = math::indexing(expr.val_op(), row_idx_expr, col_idx_expr);
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
