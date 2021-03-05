#ifndef STAN_IO_DESERIALIZER_HPP
#define STAN_IO_DESERIALIZER_HPP

#include <stan/math/rev.hpp>

namespace stan {

namespace io {

/**
 * A stream-based reader for integer, scalar, vector, matrix
 * and array data types, with Jacobian calculations.
 *
 * The template parameter <code>T</code> represents the type of
 * scalars and the values in vectors and matrices.  The only
 * requirement on the template type <code>T</code> is that a
 * double can be copied into it, as in
 *
 * <code>T t = 0.0;</code>
 *
 * This includes <code>double</code> itself and the reverse-mode
 * algorithmic differentiation class <code>stan::math::var</code>.
 *
 * <p>For transformed values, the scalar type parameter <code>T</code>
 * must support the transforming operations, such as <code>exp(x)</code>
 * for positive-bounded variables.  It must also support equality and
 * inequality tests with <code>double</code> values.
 *
 * @tparam T Basic scalar type.
 */
template <typename T>
class deserializer {
 private:
  Eigen::Map<Eigen::Matrix<T, -1, 1>> data_r_;    // map of reals.
  Eigen::Map<Eigen::Matrix<int, -1, 1>> data_i_;  // map of integers.
  size_t r_size_{0};                              // size of reals available.
  size_t i_size_{0};                              // size of integers available.
  size_t pos_r_{0};  // current position in map of reals.
  size_t pos_i_{0};  // current position in map of integers.
  /**
   * Return pointer to current scalar.
   */
  inline T& scalar_ptr() { return data_r_.coeffRef(pos_r_); }

  /**
   * Return pointer to current scalar and incriment the internal counter.
   * @param m amount to move `pos_r_` up.
   */
  inline T& scalar_ptr_increment(size_t m) {
    pos_r_ += m;
    return data_r_.coeffRef(pos_r_ - m);
  }

  /**
   * Return pointer to current integer.
   */
  inline int& int_ptr() { return data_i_.coeffRef(pos_i_); }

  /**
   * Return pointer to current integer and incriment the internal counter.
   * @param m amount to move `pos_i_` up.
   */
  inline int& int_ptr_increment(size_t m) {
    pos_i_ += m;
    return data_i_.coeffRef(pos_i_ - m);
  }
  /**
   * Check if there is anything left to read for scalars.
   */
  void check_r_capacity() const {
    if (pos_r_ >= r_size_) {
      []() STAN_COLD_PATH {
        throw std::runtime_error("no more scalars to read");
      }();
    }
  }

  /**
   * Check if there is anything left to read for integers.
   */
  void check_i_capacity() const {
    if (pos_i_ >= i_size_) {
      []() STAN_COLD_PATH {
        throw std::runtime_error("no more integers to read");
      }();
    }
  }

  template <typename S, typename K>
  using conditional_var_val_t
      = std::conditional_t<is_var_matrix<S>::value && is_var<T>::value,
                           return_var_matrix_t<K, S, K>, K>;

  template <typename S>
  using is_fp_or_ad = bool_constant<std::is_floating_point<S>::value
                                    || is_autodiff<S>::value>;

 public:
  using matrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using vector_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using row_vector_t = Eigen::Matrix<T, 1, Eigen::Dynamic>;

  using map_matrix_t = Eigen::Map<matrix_t>;
  using map_vector_t = Eigen::Map<vector_t>;
  using map_row_vector_t = Eigen::Map<row_vector_t>;

  using var_matrix_t = stan::math::var_value<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;
  using var_vector_t
      = stan::math::var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
  using var_row_vector_t
      = stan::math::var_value<Eigen::Matrix<double, 1, Eigen::Dynamic>>;

  /**
   * Construct a variable reader using the specified vectors
   * as the source of scalar and integer values for data.  This
   * class holds a reference to the specified data vectors.
   *
   * Attempting to read beyond the end of the data or integer
   * value sequences raises a runtime exception.
   *
   * @param data_r Sequence of scalar values.
   * @param data_i Sequence of integer values.
   */
  template <typename RVec, typename IntVec,
            require_all_vector_like_t<RVec, IntVec>* = nullptr>
  deserializer(RVec& data_r, IntVec& data_i)
      : data_r_(data_r.data(), data_r.size()),
        data_i_(data_i.data(), data_i.size()),
        r_size_(data_r.size()),
        i_size_(data_i.size()) {}

  /**
   * Return the number of scalars remaining to be read.
   *
   * @return Number of scalars left to read.
   */
  inline size_t available() const noexcept { return r_size_ - pos_r_; }

  /**
   * Return the number of integers remaining to be read.
   *
   * @return Number of integers left to read.
   */
  inline size_t available_i() const noexcept { return i_size_ - pos_i_; }

  /**
   * Return the next object in the sequence.
   *
   * @return Next scalar value.
   */
  template <typename Ret, require_t<is_fp_or_ad<Ret>>* = nullptr>
  auto read() {
    check_r_capacity();
    return data_r_.coeffRef(pos_r_++);
  }

  template <typename Ret, require_complex_t<Ret>* = nullptr>
  auto read() {
    check_r_capacity();
    return std::complex<T>{data_r_.coeffRef(pos_r_++),
                           data_r_.coeffRef(pos_r_++)};
  }

  /**
   * Return the next integer in the integer sequence.
   *
   * @return Next integer value.
   */
  template <typename Ret, require_integral_t<Ret>* = nullptr>
  auto read() {
    check_i_capacity();
    return data_i_.coeffRef(pos_i_++);
  }

  /**
   * Return an Eigen column vector of size `m`.
   * @tparam Size an integral type
   * @tparam Ret The type to return.
   * @param m Size of column vector.
   */
  template <typename Ret, typename Size,
            require_eigen_col_vector_t<Ret>* = nullptr,
            require_not_vt_complex<Ret>* = nullptr>
  auto read(Size m) {
    if (unlikely(m == 0)) {
      return map_vector_t(nullptr, m);
    } else {
      return map_vector_t(&scalar_ptr_increment(m), m);
    }
  }

  /**
   * Return an Eigen column vector of size `m` with inner complex type.
   * @tparam Size an integral type
   * @tparam Ret The type to return.
   * @param m Size of column vector.
   */
  template <typename Ret, typename Size,
            require_eigen_col_vector_t<Ret>* = nullptr,
            require_vt_complex<Ret>* = nullptr>
  auto read(Size m) {
    if (unlikely(m == 0)) {
      return Ret(map_vector_t(nullptr, m));
    } else {
      Ret ret(m);
      for (Eigen::Index i = 0; i < m; ++i) {
        ret.coeffRef(i) = std::complex<T>{data_r_.coeffRef(pos_r_++),
                                          data_r_.coeffRef(pos_r_++)};
      }
      return ret;
    }
  }

  /**
   * Return an Eigen row vector of size `m`.
   * @tparam Size an integral type
   * @tparam Ret The type to return.
   * @param m Size of row vector.
   */
  template <typename Ret, typename Size,
            require_eigen_row_vector_t<Ret>* = nullptr,
            require_not_vt_complex<Ret>* = nullptr>
  auto read(Size m) {
    if (unlikely(m == 0)) {
      return map_row_vector_t(nullptr, m);
    } else {
      return map_row_vector_t(&scalar_ptr_increment(m), m);
    }
  }

  /**
   * Return an Eigen row vector of size `m` with inner complex type.
   * @tparam Size an integral type
   * @tparam Ret The type to return.
   * @param m Size of row vector.
   */
  template <typename Ret, typename Size,
            require_eigen_row_vector_t<Ret>* = nullptr,
            require_vt_complex<Ret>* = nullptr>
  auto read(Size m) {
    if (unlikely(m == 0)) {
      return Ret(map_row_vector_t(nullptr, m));
    } else {
      Ret ret(m);
      for (Eigen::Index i = 0; i < m; ++i) {
        ret.coeffRef(i) = std::complex<T>{data_r_.coeffRef(pos_r_++),
                                          data_r_.coeffRef(pos_r_++)};
      }
      return ret;
    }
  }

  /**
   * Return an Eigen matrix of size `(rows, cols)`.
   * @tparam Rows Integral type.
   * @tparam Cols Integral type.
   * @tparam Ret The type to return.
   * @param rows The size of the rows of the matrix.
   * @param cols The size of the cols of the matrix.
   */
  template <typename Ret, typename Rows, typename Cols,
            require_eigen_matrix_dynamic_t<Ret>* = nullptr,
            require_not_vt_complex<Ret>* = nullptr>
  auto read(Rows rows, Cols cols) {
    if (rows == 0 || cols == 0) {
      return map_matrix_t(nullptr, rows, cols);
    } else {
      return map_matrix_t(&scalar_ptr_increment(rows * cols), rows, cols);
    }
  }

  /**
   * Return an Eigen matrix of size `(rows, cols)` with complex inner type.
   * @tparam Rows Integral type.
   * @tparam Cols Integral type.
   * @tparam Ret The type to return.
   * @param rows The size of the rows of the matrix.
   * @param cols The size of the cols of the matrix.
   */
  template <typename Ret, typename Rows, typename Cols,
            require_eigen_matrix_dynamic_t<Ret>* = nullptr,
            require_vt_complex<Ret>* = nullptr>
  auto read(Rows rows, Cols cols) {
    if (rows == 0 || cols == 0) {
      return Ret(map_matrix_t(nullptr, rows, cols));
    } else {
      Ret ret(rows, cols);
      for (Eigen::Index i = 0; i < rows * cols; ++i) {
        ret.coeffRef(i) = std::complex<T>{data_r_.coeffRef(pos_r_++),
                                          data_r_.coeffRef(pos_r_++)};
      }
      return ret;
    }
  }

  /**
   * Return a `var_value` with inner Eigen type.
   * @tparam Ret The type to return.
   * @tparam T_ Should never be set by user, set to default value of `T` for
   *  performing deduction on the class's inner type.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes A parameter pack of integral types representing the
   *  dimensions of the `var_value` matrix or vector.
   */
  template <typename Ret, typename T_ = T, typename... Sizes,
            require_var_t<T_>* = nullptr, require_var_matrix_t<Ret>* = nullptr>
  auto read(Sizes... sizes) {
    using value_t = typename std::decay_t<Ret>::value_type;
    using var_v_t = stan::math::promote_scalar_t<stan::math::var, value_t>;
    return stan::math::to_var_value(this->read<var_v_t>(sizes...));
  }

  /**
   * Return an Eigen type when the deserializers inner class is not var.
   * @tparam Ret The type to return.
   * @tparam T_ Should never be set by user, set to default value of `T` for
   *  performing deduction on the class's inner type.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes A parameter pack of integral types representing the
   *  dimensions of the `var_value` matrix or vector.
   */
  template <typename Ret, typename T_ = T, typename... Sizes,
            require_not_var_t<T_>* = nullptr,
            require_var_matrix_t<Ret>* = nullptr>
  auto read(Sizes... sizes) {
    using value_t = typename std::decay_t<Ret>::value_type;
    return this->read<value_t>(sizes...);
  }

  /**
   * Return an `std::vector`
   * @tparam Ret The type to return.
   * @tparam Size an integral type.
   * @tparam Sizes integral types.
   * @param m The size of the vector.
   * @param dims a possible set of inner container sizes passed to subsequent
   * `read` functions.
   */
  template <typename Ret, typename Size, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read(Size m, Sizes... dims) {
    if (unlikely(m == 0)) {
      return Ret();
    } else {
      std::decay_t<Ret> ret_vec;
      ret_vec.reserve(m);
      for (Size i = 0; i < m; ++i) {
        ret_vec.emplace_back(this->read<value_type_t<Ret>>(dims...));
      }
      return ret_vec;
    }
  }

  /**
   * Return the next object, checking that it's elements are
   * greater than or equal to the specified lower bound.
   *
   * <p>See <code>stan::math::check_greater_or_equal(T,double)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam LB Type of lower bound.
   * @tparam Sizes A pack of possible sizes to construct the object from.
   * @param lb Lower bound.
   * @param sizes a pack of sizes to use to construct the return.
   * @throw std::runtime_error if the scalar is less than the
   *    specified lower bound
   */
  template <typename Ret, typename LB, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_lb(const LB& lb, Sizes... sizes) {
    auto ret = this->read<Ret>(sizes...);
    using stan::math::check_greater_or_equal;
    using stan::math::value_of;
    check_greater_or_equal("io deserializer", "Lower Bound", value_of(ret),
                           value_of(lb));
    return ret;
  }

  /**
   * Overload for `std::vector` return type and a non-`std::vector` lower bound.
   * @tparam Ret The type to return.
   * @tparam LB Type of lower bound.
   * @tparam Sizes A pack of possible sizes to construct the object from.
   * @param lb Lower bound.
   * @param sizes a pack of sizes to use to construct the return.
   * @throw std::runtime_error if the scalar is less than the
   *    specified lower bound
   */
  template <typename Ret, typename LB, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_lb(const LB& lb, size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_lb<value_type_t<Ret>>(lb, sizes...));
    }
    return ret;
  }

  /**
   * Overload for `std::vector` return type and a `std::vector` lower bound.
   * @tparam Ret The type to return.
   * @tparam LB Type of lower bound.
   * @tparam Sizes A pack of possible sizes to construct the object from.
   * @param lb Lower bound.
   * @param sizes a pack of sizes to use to construct the return.
   * @throw std::runtime_error if the scalar is less than the
   *    specified lower bound
   */
  template <typename Ret, typename LB, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_lb(const std::vector<LB>& lb, size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_lb<value_type_t<Ret>>(lb[i], sizes...));
    }
    return ret;
  }

  /**
   * Return the next object transformed to have the specified
   * lower bound, possibly incrementing the specified reference with the
   * log of the absolute Jacobian determinant of the transform.
   *
   * <p>See <code>stan::math::lb_constrain(T,double,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LB Type of lower bound.
   * @tparam LP Type of log prob.
   * @tparam Sizes A pack of possible sizes to construct the object from.
   * @param lb Lower bound on result.
   * @param lp Reference to log probability variable to increment.
   * @param sizes a pack of sizes to use to construct the return.
   */
  template <typename Ret, bool Jacobian, typename LB, typename LP,
            typename... Sizes>
  auto read_lb(const LB& lb, LP& lp, Sizes... sizes) {
    if (Jacobian) {
      return stan::math::lb_constrain(this->read<Ret>(sizes...), lb, lp);
    } else {
      return stan::math::lb_constrain(this->read<Ret>(sizes...), lb);
    }
  }

  /**
   * Return the next object, checking that it is
   * less than or equal to the specified upper bound.
   *
   * <p>See <code>stan::math::check_less_or_equal(T,double)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam UB Type of upper bound.
   * @tparam Sizes A parameter pack of integral types.
   * @param ub Lower bound.
   * @param sizes A parameter pack of dimensions.
   * @throw std::runtime_error if the scalar is less than the
   *    specified lower bound
   */
  template <typename Ret, typename UB, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_ub(const UB& ub, Sizes... sizes) {
    auto ret = this->read<Ret>(sizes...);
    using stan::math::check_less_or_equal;
    using stan::math::value_of;
    check_less_or_equal("io deserializer", "Upper Bound", value_of(ret),
                        value_of(ub));
    return ret;
  }

  /**
   * Specialization for `std::vector` return type and non-`std::vector` lower
   * bound.
   * @tparam Ret The type to return.
   * @tparam UB Type of upper bound.
   * @tparam Sizes A parameter pack of integral types.
   * @param ub Lower bound.
   * @param vecsize The size of the return `std::vector`.
   * @param sizes A parameter pack of dimensions.
   * @throw std::runtime_error if the scalar is less than the
   *    specified lower bound
   */
  template <typename Ret, typename UB, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_ub(const UB& ub, size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_ub<value_type_t<Ret>>(ub, sizes...));
    }
    return ret;
  }

  /**
   * Specialization for `std::vector` return type and an `std::vector` lower
   * bound.
   * @tparam Ret The type to return.
   * @tparam UB Type of upper bound.
   * @tparam Sizes A parameter pack of integral types.
   * @param ub Lower bound.
   * @param vecsize The size of the return `std::vector`.
   * @param sizes A parameter pack of dimensions.
   * @throw std::runtime_error if the scalar is less than the
   *    specified lower bound
   */
  template <typename Ret, typename UB, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_ub(const std::vector<UB>& ub, size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_ub<value_type_t<Ret>>(ub[i], sizes...));
    }
    return ret;
  }

  /**
   * Return the next object transformed to have the specified
   * upper bound, possibly incrementing the specified reference with the
   * log of the absolute Jacobian determinant of the transform.
   *
   * <p>See <code>stan::math::ub_constrain(T,double,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam UB Type of upper bound.
   * @tparam LP Type of log prob.
   * @param ub Upper bound on result.
   * @param lp Reference to log probability variable to increment.
   * @param sizes a pack of sizes to use to construct the return.
   */
  template <typename Ret, bool Jacobian, typename UB, typename LP,
            typename... Sizes>
  auto read_ub(const UB& ub, LP& lp, Sizes... sizes) {
    if (Jacobian) {
      return stan::math::ub_constrain(this->read<Ret>(sizes...), ub, lp);
    } else {
      return stan::math::ub_constrain(this->read<Ret>(sizes...), ub);
    }
  }

  /**
   * Return the next object, checking that it's elements is between
   * the specified lower and upper bound.
   *
   * <p>See <code>stan::math::check_bounded(T, LB, UB)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam LB Type of lower bound.
   * @tparam UB Type of upper bound.
   * @tparam Sizes A parameter pack of integral types.
   * @param lb Lower bound.
   * @param ub Upper bound.
   * @param sizes The dimensions for the inner type.
   * @throw std::runtime_error if the scalar is not between the specified
   *    lower and upper bounds.
   */
  template <typename Ret, typename LB, typename UB, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_lub(const LB& lb, const UB& ub, Sizes... sizes) {
    auto ret = this->read<Ret>(sizes...);
    using stan::math::check_bounded;
    using stan::math::value_of;
    check_bounded<decltype(value_of(ret)), decltype(value_of(lb)),
                  decltype(value_of(ub))>(
        "io deserializer", "Upper and Lower Bound", value_of(ret), value_of(lb),
        value_of(ub));
    return ret;
  }

  /**
   * Specialization for `lub` constrain with an `std::vector` return type
   * and non-`std::vector` lower and upper bounds.
   *
   * @tparam Ret The type to return.
   * @tparam LB Type of lower bound.
   * @tparam UB Type of upper bound.
   * @tparam Sizes A parameter pack of integral types.
   * @param lb Lower bound.
   * @param ub Upper bound.
   * @param vecsize The size of the return `std::vector`
   * @param sizes The dimensions for the inner type.
   * @throw std::runtime_error if the scalar is not between the specified
   *    lower and upper bounds.
   */
  template <typename Ret, typename LB, typename UB, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_lub(const LB& lb, const UB& ub, size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_lub<value_type_t<Ret>>(lb, ub, sizes...));
    }
    return ret;
  }

  /**
   * Specialization for `lub` constrain with an `std::vector` return type
   * with `std::vector` lower bound and non-`std::vector` upper bound.
   *
   * @tparam Ret The type to return.
   * @tparam LB Type of lower bound.
   * @tparam UB Type of upper bound.
   * @tparam Sizes A parameter pack of integral types.
   * @param lb Lower bound.
   * @param ub Upper bound.
   * @param vecsize The size of the return `std::vector`
   * @param sizes The dimensions for the inner type.
   * @throw std::runtime_error if the scalar is not between the specified
   *    lower and upper bounds.
   */
  template <typename Ret, typename LB, typename UB, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_lub(const std::vector<LB>& lb, const UB& ub, size_t vecsize,
                Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_lub<value_type_t<Ret>>(lb[i], ub, sizes...));
    }
    return ret;
  }

  /**
   * Specialization for `lub` constrain with an `std::vector` return type
   * with a non-`std::vector` lower bound and an `std::vector` upper bound.
   *
   * @tparam Ret The type to return.
   * @tparam LB Type of lower bound.
   * @tparam UB Type of upper bound.
   * @tparam Sizes A parameter pack of integral types.
   * @param lb Lower bound.
   * @param ub Upper bound.
   * @param vecsize The size of the return `std::vector`
   * @param sizes The dimensions for the inner type.
   * @throw std::runtime_error if the scalar is not between the specified
   *    lower and upper bounds.
   */
  template <typename Ret, typename LB, typename UB, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_lub(const LB& lb, const std::vector<UB>& ub, size_t vecsize,
                Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_lub<value_type_t<Ret>>(lb, ub[i], sizes...));
    }
    return ret;
  }

  /**
   * Specialization for `lub` constrain with an `std::vector` return type
   * with `std::vector` lower bound and upper bound.
   *
   * @tparam Ret The type to return.
   * @tparam LB Type of lower bound.
   * @tparam UB Type of upper bound.
   * @tparam Sizes A parameter pack of integral types.
   * @param lb Lower bound.
   * @param ub Upper bound.
   * @param vecsize The size of the return `std::vector`
   * @param sizes The dimensions for the inner type.
   * @throw std::runtime_error if the scalar is not between the specified
   *    lower and upper bounds.
   */
  template <typename Ret, typename LB, typename UB, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_lub(const std::vector<LB>& lb, const std::vector<UB>& ub,
                size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_lub<value_type_t<Ret>>(lb[i], ub[i], sizes...));
    }
    return ret;
  }

  /**
   * Return the next object transformed to be between the
   * the specified lower and upper bounds.
   *
   * <p>See <code>stan::math::lub_constrain(T, double, double, T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LB Type of lower bound.
   * @tparam UB Type of upper bound.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lb Lower bound.
   * @param ub Upper bound.
   * @param lp Reference to log probability variable to increment.
   * @param sizes Pack of integrals to use to construct the return's type.
   */
  template <typename Ret, bool Jacobian, typename LB, typename UB, typename LP,
            typename... Sizes>
  auto read_lub(const LB& lb, const UB& ub, LP& lp, Sizes... sizes) {
    if (Jacobian) {
      return stan::math::lub_constrain(this->read<Ret>(sizes...), lb, ub, lp);
    } else {
      return stan::math::lub_constrain(this->read<Ret>(sizes...), lb, ub);
    }
  }

  /**
   * Return the next object, checking that it's elements are positive.
   *
   * <p>See <code>stan::math::check_positive(T)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes Pack of integrals to use to construct the return's type.
   *
   * @throw std::runtime_error if x is not positive
   */
  template <typename Ret, typename... Sizes>
  auto read_pos(Sizes... sizes) {
    auto ret = read<Ret>(sizes...);
    using stan::math::check_positive;
    using stan::math::value_of;
    check_positive("deserializer", "Positive Constrained", value_of(ret));
    return ret;
  }

  /**
   * Return the next object transformed to be positive, possibly
   * incrementing the specified reference with the log absolute
   * determinant of the Jacobian.
   *
   * <p>See <code>stan::math::positive_constrain(T,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp Reference to log probability variable to increment.
   * @param sizes Pack of integrals to use to construct the return's type.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_pos(LP& lp, Sizes... sizes) {
    if (Jacobian) {
      return stan::math::positive_constrain(this->read<Ret>(sizes...), lp);
    } else {
      return stan::math::positive_constrain(this->read<Ret>(sizes...));
    }
  }

  /**
   * Specialization for `std::vector` that calls itself recursivly for
   * each element.
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param vecsize Size of the return vector.
   * @param lp Reference to log probability variable to increment.
   * @param sizes Pack of integrals to use to construct the return's type.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_pos(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_pos<value_type_t<Ret>, Jacobian>(lp, sizes...));
    }
    return ret;
  }

  /**
   * This is just returns the requested type as there's nothing to check.
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Offset Type of offset.
   * @tparam Mult Type of multiplier.
   * @tparam Sizes A parameter pack of integral types.
   * @param offset Offset.
   * @param multiplier Multiplier.
   * @param sizes Pack of integrals to use to construct the return's type.
   */
  template <typename Ret, typename Offset, typename Mult, typename... Sizes>
  auto read_offset_multiplier(const Offset& offset, const Mult& multiplier,
                              Sizes... sizes) {
    return read<Ret>(sizes...);
  }

  /**
   * Return the next object transformed to have the specified offset and
   * multiplier.
   *
   * <p>See <code>stan::math::offset_multiplier_constrain(T, double,
   * double)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Offset Type of offset.
   * @tparam Mult Type of multiplier.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param offset Offset.
   * @param multiplier Multiplier.
   * @param lp Reference to log probability variable to increment.
   * @param sizes Pack of integrals to use to construct the return's type.
   * @return Next object transformed to fall between the specified
   * bounds.
   */
  template <typename Ret, bool Jacobian, typename Offset, typename Mult,
            typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_offset_multiplier(const Offset& offset, const Mult& multiplier,
                              LP& lp, Sizes... sizes) {
    using stan::math::offset_multiplier_constrain;
    if (Jacobian) {
      return offset_multiplier_constrain(this->read<Ret>(sizes...), offset,
                                         multiplier, lp);
    } else {
      return offset_multiplier_constrain(this->read<Ret>(sizes...), offset,
                                         multiplier);
    }
  }

  /**
   * Specialization of offset multiplier for `std::vector` with
   * non-`std::vector` offset and multiplier.
   *
   * <p>See <code>stan::math::offset_multiplier_constrain(T, double,
   * double)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Offset Type of offset.
   * @tparam Mult Type of multiplier.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param offset Offset.
   * @param multiplier Multiplier.
   * @param lp Reference to log probability variable to increment.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * bounds.
   */
  template <typename Ret, bool Jacobian, typename Offset, typename Mult,
            typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_offset_multiplier(const Offset& offset, const Mult& multiplier,
                              LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_offset_multiplier<value_type_t<Ret>, Jacobian>(
              offset, multiplier, lp, sizes...));
    }
    return ret;
  }

  /**
   * Specialization of offset multiplier for `std::vector` with `std::vector`
   *  offset and multiplier.
   *
   * <p>See <code>stan::math::offset_multiplier_constrain(T, double,
   * double)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Offset Type of offset.
   * @tparam Mult Type of multiplier.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param offset Offset.
   * @param multiplier Multiplier.
   * @param lp Reference to log probability variable to increment.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * bounds.
   */
  template <typename Ret, bool Jacobian, typename Offset, typename Mult,
            typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_offset_multiplier(const std::vector<Offset>& offset,
                              const std::vector<Mult>& multiplier, LP& lp,
                              const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_offset_multiplier<value_type_t<Ret>, Jacobian>(
              offset[i], multiplier[i], lp, sizes...));
    }
    return ret;
  }

  /**
   * Specialization of offset multiplier for `std::vector` with `std::vector`
   *  offset and non-`std::vector` multiplier.
   *
   * <p>See <code>stan::math::offset_multiplier_constrain(T, double,
   * double)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Offset Type of offset.
   * @tparam Mult Type of multiplier.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param offset Offset.
   * @param multiplier Multiplier.
   * @param lp Reference to log probability variable to increment.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * bounds.
   */
  template <typename Ret, bool Jacobian, typename Offset, typename Mult,
            typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_offset_multiplier(const std::vector<Offset>& offset,
                              const Mult& multiplier, LP& lp,
                              const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_offset_multiplier<value_type_t<Ret>, Jacobian>(
              offset[i], multiplier, lp, sizes...));
    }
    return ret;
  }

  /**
   * Specialization of offset multiplier for `std::vector` with
   * non-`std::vector` offset and `std::vector` multiplier.
   *
   * <p>See <code>stan::math::offset_multiplier_constrain(T, double,
   * double)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Offset Type of offset.
   * @tparam Mult Type of multiplier.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param offset Offset.
   * @param multiplier Multiplier.
   * @param lp Reference to log probability variable to increment.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * bounds.
   */
  template <typename Ret, bool Jacobian, typename Offset, typename Mult,
            typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_offset_multiplier(const Offset& offset,
                              const std::vector<Mult>& multiplier, LP& lp,
                              const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_offset_multiplier<value_type_t<Ret>, Jacobian>(
              offset, multiplier[i], lp, sizes...));
    }
    return ret;
  }

  /**
   * Return the next object, checking that it is a valid value for
   * a probability, between 0 (inclusive) and 1 (inclusive).
   *
   * <p>See <code>stan::math::check_bounded(T)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes Pack of integrals to use to construct the return's type.
   * @return Next probability value.
   */
  template <typename Ret, typename... Sizes>
  auto read_prob(Sizes... sizes) {
    auto ret = read<Ret>(sizes...);
    using stan::math::check_bounded;
    using stan::math::value_of;
    check_bounded<decltype(value_of(ret)), double, double>(
        "deserializer", "Constrained probability", value_of(ret), 0, 1);
    return ret;
  }

  /**
   * Return the next object transformed to be a probability
   * between 0 and 1, incrementing the specified reference with
   * the log of the absolute Jacobian determinant.
   *
   * <p>See <code>stan::math::prob_constrain(T)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp Reference to log probability variable to increment.
   * @param sizes Pack of integrals to use to construct the return's type.
   * @return The next scalar transformed to a probability.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_prob(LP& lp, Sizes... sizes) {
    if (Jacobian) {
      return stan::math::prob_constrain(this->read<Ret>(sizes...), lp);
    } else {
      return stan::math::prob_constrain(this->read<Ret>(sizes...));
    }
  }

  /**
   * Specialization of `read_prob` for returning an `std::vector`.
   *
   * <p>See <code>stan::math::prob_constrain(T)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp Reference to log probability variable to increment.
   * @param sizes Pack of integrals to use to construct the return's type.
   * @return The next scalar transformed to a probability.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_prob(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_prob<value_type_t<Ret>, Jacobian>(lp, sizes...));
    }
    return ret;
  }

  /**
   * Return the next object, checking that it is a valid
   * value for a correlation, between -1 (inclusive) and
   * 1 (inclusive).
   *
   * <p>See <code>stan::math::check_bounded(T)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes Pack of integrals to use to construct the return's type.
   * @return Next correlation value.
   * @throw std::runtime_error if the value is not valid
   *   for a correlation
   */
  template <typename Ret, typename... Sizes,
            require_not_matrix_t<Ret>* = nullptr,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_corr(Sizes... sizes) {
    auto ret = read<Ret>(sizes...);
    using stan::math::check_bounded;
    using stan::math::value_of;
    check_bounded<decltype(value_of(ret)), double, double>(
        "deserializer", "Correlation value", value_of(ret), -1, 1);
    return ret;
  }

  /**
   * Specialization of `read_corr` for `std::vector` returns
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes Pack of integrals to use to construct the return's type.
   * @return Next correlation value.
   * @throw std::runtime_error if the value is not valid
   *   for a correlation
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_corr(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_corr<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Specialization of `read_corr` for matrix returns
   * @tparam Ret The type to return.
   * @param k the rows and column size of the return matrix.
   * @throw std::runtime_error if the value is not valid
   *   for a correlation
   */
  template <typename Ret, require_matrix_t<Ret>* = nullptr>
  auto read_corr(size_t k) {
    auto ret = read<Ret>(k, k);
    using stan::math::check_corr_matrix;
    using stan::math::value_of;
    check_corr_matrix("stan::math::corr_matrix", "Constrained matrix",
                      value_of(ret));
    return ret;
  }

  /**
   * Return the next object transformed to be a (partial)
   * correlation between -1 and 1, incrementing the specified
   * reference with the log of the absolute Jacobian determinant.
   *
   * <p>See <code>stan::math::corr_constrain(T,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes Pack of integrals to use to construct the return's type.
   * @param lp The reference to the variable holding the log
   * probability to increment.
   * @return The next scalar transformed to a correlation.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr,
            require_not_matrix_t<Ret>* = nullptr>
  auto read_corr(LP& lp, Sizes... sizes) {
    using stan::math::corr_constrain;
    if (Jacobian) {
      return corr_constrain(this->read<Ret>(sizes...), lp);
    } else {
      return corr_constrain(this->read<Ret>(sizes...));
    }
  }

  /**
   * Return the next object transformed to be a (partial)
   * correlation between -1 and 1, incrementing the specified
   * reference with the log of the absolute Jacobian determinant.
   *
   * <p>See <code>stan::math::corr_constrain(T,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @param k Dimensions of matrix return type.
   * @param lp The reference to the variable holding the log
   * probability to increment.
   */
  template <typename Ret, bool Jacobian, typename LP,
            require_not_std_vector_t<Ret>* = nullptr,
            require_matrix_t<Ret>* = nullptr>
  auto read_corr(LP& lp, size_t k) {
    using stan::math::corr_matrix_constrain;
    if (Jacobian) {
      return corr_matrix_constrain(
          this->read<conditional_var_val_t<Ret, vector_t>>((k * (k - 1)) / 2),
          k, lp);
    } else {
      return corr_matrix_constrain(
          this->read<conditional_var_val_t<Ret, vector_t>>((k * (k - 1)) / 2),
          k);
    }
  }

  /**
   * Specialization of `read_corr` for `std::vector` return types.
   *
   * <p>See <code>stan::math::corr_constrain(T,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return The next scalar transformed to a correlation.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_corr(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_corr<value_type_t<Ret>, Jacobian>(lp, sizes...));
    }
    return ret;
  }

  /**
   * Return a unit_vector of the specified size made up of the
   * next scalars.
   *
   * <p>See <code>stan::math::check_unit_vector</code>.
   *
   * @tparam Ret The type to return.
   * @param k Size of returned unit_vector
   * @return unit_vector read from the specified size number of scalars
   * @throw std::runtime_error if the next k values is not a unit_vector
   * @throw std::invalid_argument if k is zero
   */
  template <typename Ret, require_vector_t<Ret>* = nullptr>
  auto read_unit_vector(size_t k) {
    if (unlikely(k == 0)) {
      []() STAN_COLD_PATH {
        std::string msg = "deserializer: unit vectors cannot be size 0.";
        throw std::invalid_argument(msg);
      }();
    }
    auto ret = read<Ret>(k);
    using stan::math::check_unit_vector;
    using stan::math::value_of;
    check_unit_vector("deserializer", "Unit Vector", value_of(ret));
    return ret;
  }

  /**
   * Specialization of `read_unit_vector` for an `std::vector` of unit vectors.
   *
   * <p>See <code>stan::math::check_unit_vector</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes Pack of integrals to use to construct the return's type.
   * @param vecsize The size of the return vector.
   * @return unit_vector read from the specified size number of scalars
   * @throw std::runtime_error if the next k values is not a unit_vector
   * @throw std::invalid_argument if k is zero
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_unit_vector(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_unit_vector<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Return the next unit_vector of the specified size (using one fewer
   * unconstrained scalars), incrementing the specified reference with the
   * log absolute Jacobian determinant.
   *
   * <p>See <code>stan::math::unit_vector_constrain(Eigen::Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp The reference to the variable holding the log
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return The next unit_vector of the specified size.
   * @throw std::invalid_argument if k is zero
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_unit_vector(LP& lp, Sizes... sizes) {
    using stan::math::unit_vector_constrain;
    if (Jacobian) {
      return unit_vector_constrain(this->read<Ret>(sizes...), lp);
    } else {
      return unit_vector_constrain(this->read<Ret>(sizes...));
    }
  }

  /**
   * Return the next unit_vector of the specified size (using one fewer
   * unconstrained scalars), incrementing the specified reference with the
   * log absolute Jacobian determinant.
   *
   * <p>See <code>stan::math::unit_vector_constrain(Eigen::Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return The next unit_vector of the specified size.
   * @throw std::invalid_argument if k is zero
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_unit_vector(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_unit_vector<value_type_t<Ret>, Jacobian>(lp, sizes...));
    }
    return ret;
  }

  /**
   * Return a simplex of the specified size made up of the
   * next scalars.
   *
   * <p>See <code>stan::math::check_simplex</code>.
   *
   * @tparam Ret The type to return.
   * @param k Size of returned simplex.
   * @return Simplex read from the specified size number of scalars.
   * @throw std::runtime_error if the k values is not a simplex.
   * @throw std::invalid_argument if k is zero
   */
  template <typename Ret, require_vector_t<Ret>* = nullptr,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_simplex(size_t k) {
    if (unlikely(k == 0)) {
      []() STAN_COLD_PATH {
        std::string msg = "deserializer: simplex vectors cannot be size 0.";
        throw std::invalid_argument(msg);
      }();
    }
    auto ret = read<Ret>(k);
    using stan::math::check_simplex;
    using stan::math::value_of;
    check_simplex("deserializer", "Simplex", value_of(ret));
    return ret;
  }

  /**
   * Return a simplex of the specified size made up of the
   * next scalars.
   *
   * <p>See <code>stan::math::check_simplex</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Simplex read from the specified size number of scalars.
   * @throw std::runtime_error if the k values is not a simplex.
   * @throw std::invalid_argument if k is zero
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_simplex(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_simplex<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Return the next simplex of the specified size (using one fewer
   * unconstrained scalars), incrementing the specified reference with the
   * log absolute Jacobian determinant.
   *
   * <p>See <code>stan::math::simplex_constrain(Eigen::Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp The reference to the variable holding the log
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return The next simplex of the specified size.
   * @throws std::invalid_argument if number of dimensions (`k`) is zero
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_simplex(LP& lp, Sizes... sizes) {
    using stan::math::simplex_constrain;
    if (Jacobian) {
      return simplex_constrain(this->read<Ret>(sizes...), lp);
    } else {
      return simplex_constrain(this->read<Ret>(sizes...));
    }
  }

  /**
   * Return the next simplex of the specified size (using one fewer
   * unconstrained scalars), incrementing the specified reference with the
   * log absolute Jacobian determinant.
   *
   * <p>See <code>stan::math::simplex_constrain(Eigen::Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return The next simplex of the specified size.
   * @throws std::invalid_argument if number of dimensions (`k`) is zero
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_simplex(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_simplex<value_type_t<Ret>, Jacobian>(lp, sizes...));
    }
    return ret;
  }

  /**
   * Return the next vector of specified size containing
   * values in ascending order.
   *
   * <p>See <code>stan::math::check_ordered(T)</code> for
   * behavior on failure.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Vector of positive values in ascending order.
   */
  template <typename Ret, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_ordered(Sizes... sizes) {
    auto ret = read<Ret>(sizes...);
    using stan::math::check_ordered;
    using stan::math::value_of;
    check_ordered("deserializer", "Ordered", value_of(ret));
    return ret;
  }

  /**
   * Return the next vector of specified size containing
   * values in ascending order.
   *
   * <p>See <code>stan::math::check_ordered(T)</code> for
   * behavior on failure.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Vector of positive values in ascending order.
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_ordered(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_ordered<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Return the next ordered vector of the specified
   * size, incrementing the specified reference with the log
   * absolute Jacobian of the determinant.
   *
   * <p>See <code>stan::math::ordered_constrain(Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp The reference to the variable holding the log
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next ordered vector of the specified size.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_ordered(LP& lp, Sizes... sizes) {
    using stan::math::ordered_constrain;
    if (Jacobian) {
      return ordered_constrain(this->read<Ret>(sizes...), lp);
    } else {
      return ordered_constrain(this->read<Ret>(sizes...));
    }
  }

  /**
   * Return the next ordered vector of the specified
   * size, incrementing the specified reference with the log
   * absolute Jacobian of the determinant.
   *
   * <p>See <code>stan::math::ordered_constrain(Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Sizes A parameter pack of integral types.
   * @tparam LP Type of log probability.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next ordered vector of the specified size.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_ordered(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_ordered<value_type_t<Ret>, Jacobian>(lp, sizes...));
    }
    return ret;
  }

  /**
   * Return the next vector of specified size containing
   * positive values in ascending order.
   *
   * <p>See <code>stan::math::check_positive_ordered(T)</code> for
   * behavior on failure.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Vector of positive values in ascending order.
   */
  template <typename Ret, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_positive_ordered(Sizes... sizes) {
    auto ret = read<Ret>(sizes...);
    using stan::math::check_positive_ordered;
    using stan::math::value_of;
    check_positive_ordered("deserializer", "Positive Ordered", value_of(ret));
    return ret;
  }

  /**
   * Return the next vector of specified size containing
   * positive values in ascending order.
   *
   * <p>See <code>stan::math::check_positive_ordered(T)</code> for
   * behavior on failure.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Vector of positive values in ascending order.
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_positive_ordered(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_positive_ordered<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Return the next positive_ordered vector of the specified
   * size, incrementing the specified reference with the log
   * absolute Jacobian of the determinant.
   *
   * <p>See <code>stan::math::positive_ordered_constrain(Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Sizes A parameter pack of integral types.
   * @tparam LP Type of log probability.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next positive_ordered vector of the specified size.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  auto read_positive_ordered(LP& lp, Sizes... sizes) {
    using stan::math::positive_ordered_constrain;
    if (Jacobian) {
      return positive_ordered_constrain(this->read<Ret>(sizes...), lp);
    } else {
      return positive_ordered_constrain(this->read<Ret>(sizes...));
    }
  }

  /**
   * Return the next positive_ordered vector of the specified
   * size, incrementing the specified reference with the log
   * absolute Jacobian of the determinant.
   *
   * <p>See <code>stan::math::positive_ordered_constrain(Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Sizes A parameter pack of integral types.
   * @tparam LP Type of log probability.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next positive_ordered vector of the specified size.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_positive_ordered(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_positive_ordered<value_type_t<Ret>, Jacobian>(
          lp, sizes...));
    }
    return ret;
  }

  /**
   * Return the next Cholesky factor with the specified
   * dimensionality, reading it directly without transforms.
   *
   * @tparam Ret The type to return.
   * @param M Rows of Cholesky factor
   * @param N Columns of Cholesky factor
   * @return Next Cholesky factor.
   * @throw std::domain_error if the matrix is not a valid
   * Cholesky factor.
   */
  template <typename Ret, require_matrix_t<Ret>* = nullptr>
  auto read_cholesky_factor_cov(size_t M, size_t N) {
    auto ret = read<Ret>(M, N);
    using stan::math::check_cholesky_factor;
    using stan::math::value_of;
    check_cholesky_factor("deserializer", "Cholesky Factor Cov", value_of(ret));
    return ret;
  }

  /**
   * Return the next Cholesky factor with the specified
   * dimensionality, reading it directly without transforms.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next Cholesky factor.
   * @throw std::domain_error if the matrix is not a valid
   * Cholesky factor.
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_cholesky_factor_cov(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_cholesky_factor_cov<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Return the next Cholesky factor with the specified
   * dimensionality, reading from an unconstrained vector of the
   * appropriate size, and increment the log probability reference
   * with the log Jacobian adjustment for the transform.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @param lp The reference to the variable holding the log
   * @param M Rows of Cholesky factor
   * @param N Columns of Cholesky factor
   * @return Next Cholesky factor.
   * @throw std::domain_error if the matrix is not a valid
   *    Cholesky factor.
   */
  template <typename Ret, bool Jacobian, typename LP,
            require_matrix_t<Ret>* = nullptr>
  auto read_cholesky_factor_cov(LP& lp, size_t M, size_t N) {
    if (Jacobian) {
      return stan::math::cholesky_factor_constrain(
          this->read<conditional_var_val_t<Ret, vector_t>>((N * (N + 1)) / 2
                                                           + (M - N) * N),
          M, N, lp);
    } else {
      return stan::math::cholesky_factor_constrain(
          this->read<conditional_var_val_t<Ret, vector_t>>((N * (N + 1)) / 2
                                                           + (M - N) * N),
          M, N);
    }
  }

  /**
   * Return the next Cholesky factor with the specified
   * dimensionality, reading from an unconstrained vector of the
   * appropriate size, and increment the log probability reference
   * with the log Jacobian adjustment for the transform.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam Sizes A parameter pack of integral types.
   * @tparam LP Type of log probability.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next Cholesky factor.
   * @throw std::domain_error if the matrix is not a valid
   *    Cholesky factor.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_cholesky_factor_cov(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_cholesky_factor_cov<value_type_t<Ret>, Jacobian>(
              lp, sizes...));
    }
    return ret;
  }

  /**
   * Return the next Cholesky factor for a correlation matrix with
   * the specified dimensionality, reading it directly without
   * transforms.
   *
   * @param K Rows and columns of Cholesky factor
   * @return Next Cholesky factor for a correlation matrix.
   * @throw std::domain_error if the matrix is not a valid
   * Cholesky factor for a correlation matrix.
   */
  template <typename Ret, require_matrix_t<Ret>* = nullptr>
  auto read_cholesky_factor_corr(size_t K) {
    auto ret = read<Ret>(K, K);
    using stan::math::check_cholesky_factor_corr;
    using stan::math::value_of;
    check_cholesky_factor_corr("deserializer", "Cholesky Factor Corr Matrix",
                               value_of(ret));
    return ret;
  }

  /**
   * Return the next Cholesky factor for a correlation matrix with
   * the specified dimensionality, reading it directly without
   * transforms.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next Cholesky factor for a correlation matrix.
   * @throw std::domain_error if the matrix is not a valid
   * Cholesky factor for a correlation matrix.
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_cholesky_factor_corr(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_cholesky_factor_corr<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Return the next Cholesky factor for a correlation matrix with
   * the specified dimensionality, reading from an unconstrained
   * vector of the appropriate size, and increment the log
   * probability reference with the log Jacobian adjustment for
   * the transform.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @param lp Log probability reference to increment.
   * @param K Rows and columns of Cholesky factor
   * @return Next Cholesky factor for a correlation matrix.
   * @throw std::domain_error if the matrix is not a valid
   *    Cholesky factor for a correlation matrix.
   */
  template <typename Ret, bool Jacobian, typename LP,
            require_matrix_t<Ret>* = nullptr>
  auto read_cholesky_factor_corr(LP& lp, size_t K) {
    using stan::math::cholesky_corr_constrain;
    if (Jacobian) {
      return cholesky_corr_constrain(
          this->read<conditional_var_val_t<Ret, vector_t>>((K * (K - 1)) / 2),
          K, lp);
    } else {
      return cholesky_corr_constrain(
          this->read<conditional_var_val_t<Ret, vector_t>>((K * (K - 1)) / 2),
          K);
    }
  }

  /**
   * Return the next Cholesky factor for a correlation matrix with
   * the specified dimensionality, reading from an unconstrained
   * vector of the appropriate size, and increment the log
   * probability reference with the log Jacobian adjustment for
   * the transform.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next Cholesky factor for a correlation matrix.
   * @throw std::domain_error if the matrix is not a valid
   *    Cholesky factor for a correlation matrix.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_cholesky_factor_corr(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_cholesky_factor_corr<value_type_t<Ret>, Jacobian>(
              lp, sizes...));
    }
    return ret;
  }

  /**
   * Return the next covariance matrix with the specified
   * dimensionality.
   *
   * <p>See <code>stan::math::check_cov_matrix(Matrix)</code>.
   *
   * @tparam Ret The type to return.
   * @param k Dimensionality of covariance matrix.
   * @return Next covariance matrix of the specified dimensionality.
   * @throw std::runtime_error if the matrix is not a valid
   *    covariance matrix
   */
  template <typename Ret, require_matrix_t<Ret>* = nullptr>
  auto read_cov_matrix(size_t k) {
    auto ret = read<Ret>(k, k);
    using stan::math::check_cov_matrix;
    using stan::math::value_of;
    check_cov_matrix("stan::io::cov_matrix", "Constrained matrix",
                     value_of(ret));
    return ret;
  }

  /**
   * Return the next covariance matrix with the specified
   * dimensionality.
   *
   * <p>See <code>stan::math::check_cov_matrix(Matrix)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Sizes A parameter pack of integral types.
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next covariance matrix of the specified dimensionality.
   * @throw std::runtime_error if the matrix is not a valid
   *    covariance matrix
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_cov_matrix(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(this->read_cov_matrix<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Return the next covariance matrix of the specified dimensionality,
   * incrementing the specified reference with the log absolute Jacobian
   * determinant.
   *
   * <p>See <code>stan::math::cov_matrix_constrain(Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @param lp The reference to the variable holding the log
   * @param k Dimensionality of the (square) covariance matrix.
   * @return The next covariance matrix of the specified dimensionality.
   */
  template <typename Ret, bool Jacobian, typename LP,
            require_matrix_t<Ret>* = nullptr>
  auto read_cov_matrix(LP& lp, size_t k) {
    using stan::math::cov_matrix_constrain;
    if (Jacobian) {
      return cov_matrix_constrain(
          this->read<conditional_var_val_t<Ret, vector_t>>(k
                                                           + (k * (k - 1)) / 2),
          k, lp);
    } else {
      return cov_matrix_constrain(
          this->read<conditional_var_val_t<Ret, vector_t>>(k
                                                           + (k * (k - 1)) / 2),
          k);
    }
  }

  /**
   * Return the next covariance matrix of the specified dimensionality,
   * incrementing the specified reference with the log absolute Jacobian
   * determinant.
   *
   * <p>See <code>stan::math::cov_matrix_constrain(Matrix,T&)</code>.
   *
   * @tparam Ret The type to return.
   * @tparam Jacobian Whether to increment the log of the absolute Jacobian
   * determinant of the transform.
   * @tparam LP Type of log probability.
   * @tparam Sizes A parameter pack of integral types.
   * @param lp The reference to the variable holding the log
   * @param vecsize The size of the return vector.
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return The next covariance matrix of the specified dimensionality.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  auto read_cov_matrix(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_cov_matrix<value_type_t<Ret>, Jacobian>(lp, sizes...));
    }
    return ret;
  }
};

}  // namespace io
}  // namespace stan

#endif
