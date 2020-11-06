#ifndef STAN_MODEL_INDEXING_LVALUE_VARMAT_HPP
#define STAN_MODEL_INDEXING_LVALUE_VARMAT_HPP

#include <stan/math/rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/lvalue.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <type_traits>
#include <vector>

namespace stan {

namespace model {

/**
 * Indexing Notes:
 * The different index types:
 * index_uni - A single cell.
 * index_multi - Access multiple cells.
 * index_omni - A no-op for all indices along a dimension.
 * index_min - index from min:N
 * index_max - index from 1:max
 * index_min_max - index from min:max
 * nil_index_list - no-op
 * The order of the overloads are
 * vector / row_vector:
 *  - all index overloads
 * matrix:
 *  - all row index overloads
 *    - Assigns a subset of rows.
 *  - column/row overloads
 *    - overload on both the row and column indices.
 *  - column overloads
 *    - These take a subset of columns and then call the row slice assignment
 *       over the column subset.
 * Std vector:
 *  - single element and elementwise overloads
 *  - General overload for nested std vectors.
 */


/**
 * Assign to a single element of an Eigen Vector.
 *
 * Types: vector[uni] <- scalar
 *
 * @tparam Vec Eigen type with either dynamic rows or columns, but not both.
 * @tparam U Type of value (must be assignable to T).
 * @param[in] x Vector variable to be assigned.
 * @param[in] idxs index holding which cell to assign to.
 * @param[in] y Value to assign.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename VarVec, typename U, require_var_vector_t<VarVec>* = nullptr,
          require_var_t<U>* = nullptr>
inline void assign(VarVec&& x,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const U& y, const char* name = "ANON", int depth = 0) {
  stan::math::check_range("var_vector[uni] assign range", name, x.size(),
                          idxs.head_.n_);
  const auto coeff_idx = idxs.head_.n_ - 1;
  double prev_val = x.val().coeffRef(coeff_idx);
  x.vi_->val_.coeffRef(coeff_idx) = y.val();
  stan::math::reverse_pass_callback([x_vi = x.vi_, y, coeff_idx, prev_val]() mutable {
    x_vi->val_.coeffRef(coeff_idx) = prev_val;
    y.adj() += x_vi->adj_.coeffRef(coeff_idx);
    x_vi->adj_.coeffRef(coeff_idx) = 0.0;
  });
}

/**
 * Assign to a non-contiguous subset of elements in a vector.
 *
 * Types:  vector[multi] <- vector
 *
 * @tparam Vec1 Eigen type with either dynamic rows or columns, but not both.
 * @tparam Vec2 Eigen type with either dynamic rows or columns, but not both.
 * @param[in] x Vector to be assigned.
 * @param[in] idxs Index holding an `std::vector` of cells to assign to.
 * @param[in] y Value vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2,
          require_all_var_vector_t<Vec1, Vec2>* = nullptr>
inline void assign(Vec1&& x,
                   const cons_index_list<index_multi, nil_index_list>& idxs,
                   const Vec2& y, const char* name = "ANON", int depth = 0) {
  stan::math::check_size_match("vector[multi] assign sizes", "lhs",
                               idxs.head_.ns_.size(), name, y.size());
  const auto x_size = x.size();
  arena_t<Eigen::Matrix<double, -1, 1>> prev_vals(idxs.head_.ns_.size());
  arena_t<std::vector<int>> multi_idx(idxs.head_.ns_.size());
  for (int n = 0; n < y.size(); ++n) {
    const auto coeff_idx = idxs.head_.ns_[n] - 1;
    stan::math::check_range("vector[multi] assign range", name, x_size,
                            coeff_idx + 1);
    multi_idx[n] = coeff_idx;
    prev_vals.coeffRef(n) = x.vi_->val_.coeffRef(coeff_idx);
    x.vi_->val_.coeffRef(coeff_idx) = y.vi_->val_.coeff(n);
  }
  stan::math::reverse_pass_callback([x_vi = x.vi_, y_vi = y.vi_, prev_vals, multi_idx]() mutable {
    for (Eigen::Index i = 0; i < multi_idx.size(); ++i) {
      const auto coeff_idx = multi_idx[i];
      x_vi->val_.coeffRef(coeff_idx) = prev_vals.coeffRef(i);
      y_vi->adj_.coeffRef(i) += x_vi->adj_.coeffRef(coeff_idx);
      x_vi->adj_.coeffRef(coeff_idx) = 0.0;
    }
  });
}

}  // namespace model
}  // namespace stan
#endif
