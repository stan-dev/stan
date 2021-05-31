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
  Eigen::Map<Eigen::Matrix<T, -1, 1>> map_r_;    // map of reals.
  Eigen::Map<Eigen::Matrix<int, -1, 1>> map_i_;  // map of integers.
  size_t r_size_{0};                             // size of reals available.
  size_t i_size_{0};                             // size of integers available.
  size_t pos_r_{0};  // current position in map of reals.
  size_t pos_i_{0};  // current position in map of integers.

  /**
   * Return reference to current scalar and increment the internal counter.
   * @param m amount to move `pos_r_` up.
   */
  inline T& scalar_ptr_increment(size_t m) {
    pos_r_ += m;
    return map_r_.coeffRef(pos_r_ - m);
  }

  /**
   * Check there are at least m reals left to read
   * @param m Number of reals to read
   * @throws std::runtime_error if there aren't at least m reals left
   */
  void check_r_capacity(size_t m) const {
    STAN_NO_RANGE_CHECKS_RETURN;
    if (pos_r_ + m > r_size_) {
      []() STAN_COLD_PATH {
        throw std::runtime_error("no more scalars to read");
      }();
    }
  }

  /**
   * Check there are at least m integers left to read
   * @param m Number of integers to read
   * @throws std::runtime_error if there aren't at least m integers left
   */
  void check_i_capacity(size_t m) const {
    STAN_NO_RANGE_CHECKS_RETURN;
    if (pos_i_ + m > i_size_) {
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
      : map_r_(data_r.data(), data_r.size()),
        map_i_(data_i.data(), data_i.size()),
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
  inline auto read() {
    check_r_capacity(1);
    return map_r_.coeffRef(pos_r_++);
  }

  /**
   * Construct a complex variable from the next two reals in the sequence
   *
   * @return Next complex value
   */
  template <typename Ret, require_complex_t<Ret>* = nullptr>
  inline auto read() {
    check_r_capacity(2);
    auto real = scalar_ptr_increment(1);
    auto imag = scalar_ptr_increment(1);
    return std::complex<T>{real, imag};
  }

  /**
   * Return the next integer in the integer sequence.
   *
   * @return Next integer value.
   */
  template <typename Ret, require_integral_t<Ret>* = nullptr>
  inline auto read() {
    check_i_capacity(1);
    return map_i_.coeffRef(pos_i_++);
  }

  /**
   * Return an Eigen column vector of size `m`.
   * @tparam Ret The type to return.
   * @param m Size of column vector.
   */
  template <typename Ret, require_eigen_col_vector_t<Ret>* = nullptr,
            require_not_vt_complex<Ret>* = nullptr>
  inline auto read(Eigen::Index m) {
    if (unlikely(m == 0)) {
      return map_vector_t(nullptr, m);
    } else {
      check_r_capacity(m);
      return map_vector_t(&scalar_ptr_increment(m), m);
    }
  }

  /**
   * Return an Eigen column vector of size `m` with inner complex type.
   * @tparam Ret The type to return.
   * @param m Size of column vector.
   */
  template <typename Ret, require_eigen_col_vector_t<Ret>* = nullptr,
            require_vt_complex<Ret>* = nullptr>
  inline auto read(Eigen::Index m) {
    if (unlikely(m == 0)) {
      return Ret(map_vector_t(nullptr, m));
    } else {
      check_r_capacity(2 * m);
      Ret ret(m);
      for (Eigen::Index i = 0; i < m; ++i) {
        auto real = scalar_ptr_increment(1);
        auto imag = scalar_ptr_increment(1);
        ret.coeffRef(i) = std::complex<T>{real, imag};
      }
      return ret;
    }
  }

  /**
   * Return an Eigen row vector of size `m`.
   * @tparam Ret The type to return.
   * @param m Size of row vector.
   */
  template <typename Ret, require_eigen_row_vector_t<Ret>* = nullptr,
            require_not_vt_complex<Ret>* = nullptr>
  inline auto read(Eigen::Index m) {
    if (unlikely(m == 0)) {
      return map_row_vector_t(nullptr, m);
    } else {
      check_r_capacity(m);
      return map_row_vector_t(&scalar_ptr_increment(m), m);
    }
  }

  /**
   * Return an Eigen row vector of size `m` with inner complex type.
   * @tparam Ret The type to return.
   * @param m Size of row vector.
   */
  template <typename Ret, require_eigen_row_vector_t<Ret>* = nullptr,
            require_vt_complex<Ret>* = nullptr>
  inline auto read(Eigen::Index m) {
    if (unlikely(m == 0)) {
      return Ret(map_row_vector_t(nullptr, m));
    } else {
      check_r_capacity(2 * m);
      Ret ret(m);
      for (Eigen::Index i = 0; i < m; ++i) {
        auto real = scalar_ptr_increment(1);
        auto imag = scalar_ptr_increment(1);
        ret.coeffRef(i) = std::complex<T>{real, imag};
      }
      return ret;
    }
  }

  /**
   * Return an Eigen matrix of size `(rows, cols)`.
   * @tparam Ret The type to return.
   * @param rows The size of the rows of the matrix.
   * @param cols The size of the cols of the matrix.
   */
  template <typename Ret, require_eigen_matrix_dynamic_t<Ret>* = nullptr,
            require_not_vt_complex<Ret>* = nullptr>
  inline auto read(Eigen::Index rows, Eigen::Index cols) {
    if (rows == 0 || cols == 0) {
      return map_matrix_t(nullptr, rows, cols);
    } else {
      check_r_capacity(rows * cols);
      return map_matrix_t(&scalar_ptr_increment(rows * cols), rows, cols);
    }
  }

  /**
   * Return an Eigen matrix of size `(rows, cols)` with complex inner type.
   * @tparam Ret The type to return.
   * @param rows The size of the rows of the matrix.
   * @param cols The size of the cols of the matrix.
   */
  template <typename Ret, require_eigen_matrix_dynamic_t<Ret>* = nullptr,
            require_vt_complex<Ret>* = nullptr>
  inline auto read(Eigen::Index rows, Eigen::Index cols) {
    if (rows == 0 || cols == 0) {
      return Ret(map_matrix_t(nullptr, rows, cols));
    } else {
      check_r_capacity(2 * rows * cols);
      Ret ret(rows, cols);
      for (Eigen::Index i = 0; i < rows * cols; ++i) {
        auto real = scalar_ptr_increment(1);
        auto imag = scalar_ptr_increment(1);
        ret.coeffRef(i) = std::complex<T>{real, imag};
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
  inline auto read(Sizes... sizes) {
    using stan::math::promote_scalar_t;
    using var_v_t = promote_scalar_t<stan::math::var, value_type_t<Ret>>;
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
  inline auto read(Sizes... sizes) {
    return this->read<value_type_t<Ret>>(sizes...);
  }

  /**
   * Return an `std::vector`
   * @tparam Ret The type to return.
   * @tparam Sizes integral types.
   * @param m The size of the vector.
   * @param dims a possible set of inner container sizes passed to subsequent
   * `read` functions.
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr,
            require_not_same_t<value_type_t<Ret>, T>* = nullptr>
  inline auto read(Eigen::Index m, Sizes... dims) {
    if (unlikely(m == 0)) {
      return std::decay_t<Ret>();
    } else {
      std::decay_t<Ret> ret_vec;
      ret_vec.reserve(m);
      for (size_t i = 0; i < m; ++i) {
        ret_vec.emplace_back(this->read<value_type_t<Ret>>(dims...));
      }
      return ret_vec;
    }
  }

  /**
   * Return an `std::vector` of scalars
   * @tparam Ret The type to return.
   * @tparam Sizes integral types.
   * @param m The size of the vector.
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr,
            require_same_t<value_type_t<Ret>, T>* = nullptr>
  inline auto read(Eigen::Index m) {
    if (unlikely(m == 0)) {
      return std::decay_t<Ret>();
    } else {
      check_r_capacity(m);
      const auto* start_pos = &this->map_r_.coeffRef(this->pos_r_);
      const auto* end_pos = &this->map_r_.coeffRef(this->pos_r_ + m);
      this->pos_r_ += m;
      return std::decay_t<Ret>(start_pos, end_pos);
    }
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
  inline auto read_constrain_lb(const LB& lb, LP& lp, Sizes... sizes) {
    if (Jacobian) {
      return stan::math::lb_constrain(this->read<Ret>(sizes...), lb, lp);
    } else {
      return stan::math::lb_constrain(this->read<Ret>(sizes...), lb);
    }
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
  inline auto read_constrain_ub(const UB& ub, LP& lp, Sizes... sizes) {
    if (Jacobian) {
      return stan::math::ub_constrain(this->read<Ret>(sizes...), ub, lp);
    } else {
      return stan::math::ub_constrain(this->read<Ret>(sizes...), ub);
    }
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
  inline auto read_constrain_lub(const LB& lb, const UB& ub, LP& lp,
                                 Sizes... sizes) {
    if (Jacobian) {
      return stan::math::lub_constrain(this->read<Ret>(sizes...), lb, ub, lp);
    } else {
      return stan::math::lub_constrain(this->read<Ret>(sizes...), lb, ub);
    }
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
            typename LP, typename... Sizes>
  inline auto read_constrain_offset_multiplier(const Offset& offset,
                                               const Mult& multiplier, LP& lp,
                                               Sizes... sizes) {
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
  inline auto read_constrain_unit_vector(LP& lp, Sizes... sizes) {
    using stan::math::unit_vector_constrain;
    if (Jacobian) {
      return math::eval(unit_vector_constrain(this->read<Ret>(sizes...), lp));
    } else {
      return math::eval(unit_vector_constrain(this->read<Ret>(sizes...)));
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
  inline auto read_constrain_unit_vector(LP& lp, const size_t vecsize,
                                         Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_constrain_unit_vector<value_type_t<Ret>, Jacobian>(
              lp, sizes...));
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
  template <typename Ret, bool Jacobian, typename LP,
            require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_constrain_simplex(LP& lp, size_t size) {
    using stan::math::simplex_constrain;
    stan::math::check_positive("read_simplex", "size", size);
    if (Jacobian) {
      return simplex_constrain(this->read<Ret>(size - 1), lp);
    } else {
      return simplex_constrain(this->read<Ret>(size - 1));
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
  inline auto read_constrain_simplex(LP& lp, const size_t vecsize,
                                     Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_constrain_simplex<value_type_t<Ret>, Jacobian>(lp,
                                                                    sizes...));
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
  inline auto read_constrain_ordered(LP& lp, Sizes... sizes) {
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
  inline auto read_constrain_ordered(LP& lp, const size_t vecsize,
                                     Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_constrain_ordered<value_type_t<Ret>, Jacobian>(lp,
                                                                    sizes...));
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
   * @param sizes Pack of integrals to use to construct the return's type.
   * probability to increment.
   * @return Next positive_ordered vector of the specified size.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes,
            require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_constrain_positive_ordered(LP& lp, Sizes... sizes) {
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
  inline auto read_constrain_positive_ordered(LP& lp, const size_t vecsize,
                                              Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_constrain_positive_ordered<value_type_t<Ret>, Jacobian>(
              lp, sizes...));
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
  inline auto read_constrain_cholesky_factor_cov(LP& lp, Eigen::Index M,
                                                 Eigen::Index N) {
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
  inline auto read_constrain_cholesky_factor_cov(LP& lp, const size_t vecsize,
                                                 Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_constrain_cholesky_factor_cov<value_type_t<Ret>, Jacobian>(
              lp, sizes...));
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
  inline auto read_constrain_cholesky_factor_corr(LP& lp, Eigen::Index K) {
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
  inline auto read_constrain_cholesky_factor_corr(LP& lp, const size_t vecsize,
                                                  Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_constrain_cholesky_factor_corr<value_type_t<Ret>,
                                                    Jacobian>(lp, sizes...));
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
  inline auto read_constrain_cov_matrix(LP& lp, Eigen::Index k) {
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
  auto read_constrain_cov_matrix(LP& lp, const size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_constrain_cov_matrix<value_type_t<Ret>, Jacobian>(
              lp, sizes...));
    }
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
   * @param k Dimensions of matrix return type.
   * @param lp The reference to the variable holding the log
   * probability to increment.
   */
  template <typename Ret, bool Jacobian, typename LP,
            require_not_std_vector_t<Ret>* = nullptr,
            require_matrix_t<Ret>* = nullptr>
  inline auto read_constrain_corr_matrix(LP& lp, Eigen::Index k) {
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
  inline auto read_constrain_corr_matrix(LP& lp, const size_t vecsize,
                                         Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          this->read_constrain_corr_matrix<value_type_t<Ret>, Jacobian>(
              lp, sizes...));
    }
    return ret;
  }

  /**
   * Read a serialized lower bounded variable and unconstrain it
   *
   * @tparam Ret Type of output
   * @tparam L Type of lower bound
   * @tparam Sizes Types of dimensions of output
   * @param lb Lower bound
   * @param sizes dimensions
   * @return Unconstrained variable
   */
  template <typename Ret, typename L, typename... Sizes>
  inline auto read_free_lb(const L& lb, Sizes... sizes) {
    return stan::math::lb_free(this->read<Ret>(sizes...), lb);
  }

  /**
   * Read a serialized lower bounded variable and unconstrain it
   *
   * @tparam Ret Type of output
   * @tparam U Type of upper bound
   * @tparam Sizes Types of dimensions of output
   * @param ub Upper bound
   * @param sizes dimensions
   * @return Unconstrained variable
   */
  template <typename Ret, typename U, typename... Sizes>
  inline auto read_free_ub(const U& ub, Sizes... sizes) {
    return stan::math::ub_free(this->read<Ret>(sizes...), ub);
  }

  /**
   * Read a serialized bounded variable and unconstrain it
   *
   * @tparam Ret Type of output
   * @tparam L Type of lower bound
   * @tparam U Type of upper bound
   * @tparam Sizes Types of dimensions of output
   * @param lb Lower bound
   * @param ub Upper bound
   * @param sizes dimensions
   * @return Unconstrained variable
   */
  template <typename Ret, typename L, typename U, typename... Sizes>
  inline auto read_free_lub(const L& lb, const U& ub, Sizes... sizes) {
    return stan::math::lub_free(this->read<Ret>(sizes...), lb, ub);
  }

  /**
   * Read a serialized offset-multiplied variable and unconstrain it
   *
   * @tparam Ret Type of output
   * @tparam O Type of offset
   * @tparam M Type of multiplier
   * @tparam Sizes Types of dimensions of output
   * @param offset Offset
   * @param multiplier Multiplier
   * @param sizes dimensions
   * @return Unconstrained variable
   */
  template <typename Ret, typename O, typename M, typename... Sizes>
  inline auto read_free_offset_multiplier(const O& offset, const M& multiplier,
                                          Sizes... sizes) {
    return stan::math::offset_multiplier_free(this->read<Ret>(sizes...), offset,
                                              multiplier);
  }

  /**
   * Read a serialized unit vector and unconstrain it
   *
   * @tparam Ret Type of output
   * @return Unconstrained vector
   */
  template <typename Ret, require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_free_unit_vector(size_t size) {
    return stan::math::unit_vector_free(this->read<Ret>(size));
  }

  /**
   * Read serialized unit vectors and unconstrain them
   *
   * @tparam Ret Type of output
   * @tparam Sizes Types of dimensions of output
   * @param vecsize Vector size
   * @param sizes dimensions
   * @return Unconstrained vectors
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read_free_unit_vector(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(read_free_unit_vector<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Read a serialized simplex and unconstrain it
   *
   * @tparam Ret Type of output
   * @return Unconstrained vector
   */
  template <typename Ret, require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_free_simplex(size_t size) {
    return stan::math::simplex_free(this->read<Ret>(size));
  }

  /**
   * Read serialized simplices and unconstrain them
   *
   * @tparam Ret Type of output
   * @tparam Sizes Types of dimensions of output
   * @param vecsize Vector size
   * @param sizes dimensions
   * @return Unconstrained vectors
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read_free_simplex(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(read_free_simplex<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Read a serialized ordered and unconstrain it
   *
   * @tparam Ret Type of output
   * @return Unconstrained vector
   */
  template <typename Ret, require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_free_ordered(size_t size) {
    return stan::math::ordered_free(this->read<Ret>(size));
  }

  /**
   * Read serialized ordered vectors and unconstrain them
   *
   * @tparam Ret Type of output
   * @tparam Sizes Types of dimensions of output
   * @param vecsize Vector size
   * @param sizes dimensions
   * @return Unconstrained vectors
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read_free_ordered(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(read_free_ordered<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Read a serialized positive ordered vector and unconstrain it
   *
   * @tparam Ret Type of output
   * @return Unconstrained vector
   */
  template <typename Ret, require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_free_positive_ordered(size_t size) {
    return stan::math::positive_ordered_free(this->read<Ret>(size));
  }

  /**
   * Read serialized positive ordered vectors and unconstrain them
   *
   * @tparam Ret Type of output
   * @tparam Sizes Types of dimensions of output
   * @param vecsize Vector size
   * @param sizes dimensions
   * @return Unconstrained vectors
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read_free_positive_ordered(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(read_free_positive_ordered<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Read a serialized covariance matrix cholesky factor and unconstrain it
   *
   * @tparam Ret Type of output
   * @param M Rows of matrix
   * @param N Cols of matrix
   * @return Unconstrained matrix
   */
  template <typename Ret>
  inline auto read_free_cholesky_factor_cov(Eigen::Index M, Eigen::Index N) {
    return stan::math::cholesky_factor_free(this->read<Ret>(M, N));
  }

  /**
   * Read serialized covariance matrix cholesky factors and unconstrain them
   *
   * @tparam Ret Type of output
   * @tparam Sizes Types of dimensions of output
   * @param vecsize Vector size
   * @param sizes dimensions
   * @return Unconstrained matrices
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read_free_cholesky_factor_cov(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          read_free_cholesky_factor_cov<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Read a serialized covariance matrix cholesky factor and unconstrain it
   *
   * @tparam Ret Type of output
   * @param M Rows/Cols of matrix
   * @return Unconstrained matrix
   */
  template <typename Ret, require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_free_cholesky_factor_corr(size_t M) {
    return stan::math::cholesky_corr_free(this->read<Ret>(M, M));
  }

  /**
   * Read serialized correlation matrix cholesky factors and unconstrain them
   *
   * @tparam Ret Type of output
   * @tparam Sizes Types of dimensions of output
   * @param vecsize Vector size
   * @param sizes dimensions
   * @return Unconstrained matrices
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read_free_cholesky_factor_corr(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(
          read_free_cholesky_factor_corr<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Read a serialized covariance matrix cholesky factor and unconstrain it
   *
   * @tparam Ret Type of output
   * @param M Rows/Cols of matrix
   * @return Unconstrained matrix
   */
  template <typename Ret, require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_free_cov_matrix(size_t M) {
    return stan::math::cov_matrix_free(this->read<Ret>(M, M));
  }

  /**
   * Read serialized covariance matrices and unconstrain them
   *
   * @tparam Ret Type of output
   * @tparam Sizes Types of dimensions of output
   * @param vecsize Vector size
   * @param sizes dimensions
   * @return Unconstrained matrices
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read_free_cov_matrix(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(read_free_cov_matrix<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }

  /**
   * Read a serialized covariance matrix cholesky factor and unconstrain it
   *
   * @tparam Ret Type of output
   * @param M Rows/Cols of matrix
   * @return Unconstrained matrix
   */
  template <typename Ret, require_not_std_vector_t<Ret>* = nullptr>
  inline auto read_free_corr_matrix(size_t M) {
    return stan::math::corr_matrix_free(this->read<Ret>(M, M));
  }

  /**
   * Read serialized correlation matrices and unconstrain them
   *
   * @tparam Ret Type of output
   * @tparam Sizes Types of dimensions of output
   * @param vecsize Vector size
   * @param sizes dimensions
   * @return Unconstrained matrices
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read_free_corr_matrix(size_t vecsize, Sizes... sizes) {
    std::decay_t<Ret> ret;
    ret.reserve(vecsize);
    for (size_t i = 0; i < vecsize; ++i) {
      ret.emplace_back(read_free_corr_matrix<value_type_t<Ret>>(sizes...));
    }
    return ret;
  }
};

}  // namespace io
}  // namespace stan

#endif
