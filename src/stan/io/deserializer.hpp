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
 Eigen::Map<const Eigen::Matrix<T, -1, 1>> data_r_;
 Eigen::Map<const Eigen::Matrix<int, -1, 1>> data_i_;
 size_t r_size_{0};
 size_t i_size_{0};
 size_t pos_r_{0};
 size_t pos_i_{0};

 inline const T &scalar_ptr() { return data_r_.coeffRef(pos_r_); }

 inline const T& scalar_ptr_increment(size_t m) {
   pos_r_ += m;
   return data_r_.coeffRef(pos_r_ - m);
 }

 inline const int &int_ptr() { return data_i_.coeffRef(pos_i_); }

 inline const int &int_ptr_increment(size_t m) {
   pos_i_ += m;
   return data_i_.coeffRef(pos_i_ - m);
 }
 void check_r_capacity() const {
   if (pos_r_ >= r_size_) {
     []() STAN_COLD_PATH {
       throw std::runtime_error("no more scalars to read");
     }();
   }
 }

 void check_i_capacity() const {
   if (pos_i_ >= i_size_) {
     []() STAN_COLD_PATH {
       throw std::runtime_error("no more integers to read");
     }();
   }
 }

public:
 using matrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
 using vector_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
 using row_vector_t = Eigen::Matrix<T, 1, Eigen::Dynamic>;

 using map_matrix_t = Eigen::Map<const matrix_t>;
 using map_vector_t = Eigen::Map<const vector_t>;
 using map_row_vector_t = Eigen::Map<const row_vector_t>;

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
           require_all_vector_like_t<RVec, IntVec> * = nullptr>
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

 template <typename Ret, require_any_t<std::is_floating_point<Ret>, is_autodiff<Ret>>* = nullptr>
 auto read() {
   check_r_capacity();
   return data_r_.coeffRef(pos_r_++);
 }

 template <typename Ret, require_complex_t<Ret>* = nullptr>
 auto read() {
   check_r_capacity();
   return std::complex<T>(data_r_.coeffRef(pos_r_++), data_r_.coeffRef(pos_r_++));
 }

 template <typename Ret, require_integral_t<Ret>* = nullptr>
 auto read() {
   check_i_capacity();
   return data_i_.coeffRef(pos_i_++);
 }

 template <typename Ret, typename Size, require_eigen_col_vector_t<Ret>* = nullptr>
 auto read(Size m) {
   if (unlikely(m == 0)) {
     return map_vector_t(nullptr, m);
   } else {
     return map_vector_t(&scalar_ptr_increment(m), m);
   }
 }

 template <typename Ret, typename Size, require_eigen_row_vector_t<Ret>* = nullptr>
 auto read(Size m) {
   if (unlikely(m == 0)) {
     return map_row_vector_t(nullptr, m);
   } else {
     return map_row_vector_t(&scalar_ptr_increment(m), m);
   }
 }

 template <typename Ret, typename Rows, typename Cols, require_eigen_matrix_dynamic_t<Ret>* = nullptr>
 auto read(Rows rows, Cols cols) {
   if (rows == 0 || cols == 0) {
     return map_matrix_t(nullptr, rows, cols);
   } else {
     return map_matrix_t(&scalar_ptr_increment(rows * cols), rows, cols);
   }
 }

 template <typename Ret, typename T_ = T, typename... Sizes,
  require_var_t<T_>* = nullptr,
  require_var_matrix_t<Ret>* = nullptr>
 auto read(Sizes... sizes) {
   using value_t = typename Ret::value_type;
   return stan::math::to_var_value(this->read<value_t>(sizes...));
 }

 template <typename Ret, typename T_ = T, typename... Sizes,
 require_not_var_t<T_>* = nullptr,
 require_var_matrix_t<Ret>* = nullptr>
 auto read(Sizes... sizes) {
   using value_t = typename Ret::value_type;
   return this->read<value_t>(sizes...);
 }

 template <typename Ret, typename Size, typename... Sizes, require_std_vector_t<Ret>* = nullptr>
 inline auto read(Size m, Sizes... dims) {
   if (unlikely(m == 0)) {
     return Ret();
   } else {
     using ret_value_type = value_type_t<Ret>;
     Ret ret_vec;
     ret_vec.reserve(m);
     for (Size i = 0; i < m; ++i) {
       ret_vec.emplace_back(this->read<ret_value_type>(dims...));
     }
     return ret_vec;
   }
 }

 template <typename Ret, typename LB, typename... Sizes>
 auto read_lb(const LB& lb, Sizes... sizes) {
   using stan::math::check_greater_or_equal;
   auto ret = this->read<Ret>(sizes...);
   check_greater_or_equal("io deserializer", "Lower Bound", ret, lb);
   return ret;
 }


 template <typename Ret, bool Jacobian, typename LB, typename LP, typename... Sizes>
 auto read_lb_constrain(const LB& lb, LP& lp, Sizes... sizes) {
   if (Jacobian) {
     return stan::math::lb_constrain(this->read<Ret>(sizes...), lb, lp);
   } else {
     return stan::math::lb_constrain(this->read<Ret>(sizes...), lb);
   }
 }

 template <typename Ret, typename UB, typename... Sizes>
 auto read_ub(const UB& ub, Sizes... sizes) {
   using stan::math::check_less_or_equal;
   auto ret = this->read<Ret>(sizes...);
   check_less_or_equal("io deserializer", "Upper Bound", ret, ub);
   return ret;
 }


 template <typename Ret, bool Jacobian, typename UB, typename LP, typename... Sizes>
 auto read_ub_constrain(const UB& ub, LP& lp, Sizes... sizes) {
   if (Jacobian) {
     return stan::math::ub_constrain(this->read<Ret>(sizes...), ub, lp);
   } else {
     return stan::math::ub_constrain(this->read<Ret>(sizes...), ub);
   }
 }

 template <typename Ret, typename LB, typename UB, typename... Sizes>
 auto read_lub(const LB& lb, const UB& ub, Sizes... sizes) {
   using stan::math::check_bounded;
   auto ret = this->read<Ret>(sizes...);
   check_bounded<Ret, LB, UB>("io deserializer", "Upper and Lower Bound", ret, lb, ub);
   return ret;
 }


 template <typename Ret, bool Jacobian, typename LB, typename UB, typename LP, typename... Sizes>
 auto read_lub_constrain(const LB& lb, const UB& ub, LP& lp, Sizes... sizes) {
   if (Jacobian) {
     return stan::math::lub_constrain(this->read<Ret>(sizes...), lb, ub, lp);
   } else {
     return stan::math::lub_constrain(this->read<Ret>(sizes...), lb, ub);
   }
 }

};


}
}

#endif
