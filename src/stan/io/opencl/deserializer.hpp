#ifndef STAN_IO_OPENCL_DESERIALIZER_HPP
#define STAN_IO_OPENCL_DESERIALIZER_HPP

#ifdef STAN_OPENCL

#include <stan/io/deserializer.hpp>
#include <stan/io/opencl/utils.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/opencl/rev/vari.hpp>
#include <stan/math/opencl/prim/constraint/lb_constrain.hpp>
#include <stan/math/opencl/prim/constraint/ub_constrain.hpp>
#include <stan/math/opencl/prim/constraint/lub_constrain.hpp>
#include <stan/math/opencl/prim/constraint/offset_multiplier_constrain.hpp>
#include <stan/math/opencl/prim/constraint/unit_vector_constrain.hpp>
#include <stan/math/opencl/rev/constraint/lb_constrain.hpp>
#include <stan/math/opencl/rev/constraint/ub_constrain.hpp>
#include <stan/math/opencl/rev/constraint/lub_constrain.hpp>
#include <stan/math/opencl/rev/constraint/offset_multiplier_constrain.hpp>
#include <stan/math/opencl/rev/constraint/unit_vector_constrain.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/rev/core.hpp>

#include <Eigen/Dense>
#include <CL/opencl.hpp>

#include <complex>
#include <vector>
#include <type_traits>

namespace stan {
namespace io {

/**
 * OpenCL deserializer specialization for matrix_cl buffers (prim).
 */
template <>
class deserializer<stan::math::matrix_cl<double>> {
 using mat_t = stan::math::matrix_cl<double>;
 private:
  const mat_t& data_r_;
  Eigen::Map<const Eigen::Matrix<int, -1, 1>> map_i_;
  size_t r_size_{0};
  size_t i_size_{0};
  size_t pos_r_{0};
  size_t pos_i_{0};
  size_t align_elems_{1};

  /**
   * Check there are at least m reals left to read.
   *
   * @param m Number of real elements to read.
   * @throws std::runtime_error if there are insufficient elements.
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
   * Check there are at least m integers left to read.
   *
   * @param m Number of integer elements to read.
   * @throws std::runtime_error if there are insufficient elements.
   */
  void check_i_capacity(size_t m) const {
    STAN_NO_RANGE_CHECKS_RETURN;
    if (pos_i_ + m > i_size_) {
      []() STAN_COLD_PATH {
        throw std::runtime_error("no more integers to read");
      }();
    }
  }

  /**
   * Align the real position to the next aligned element offset.
   */
  inline void align_pos() { pos_r_ = internal::round_up(pos_r_, align_elems_); }

  /**
   * Read a block as a matrix_cl subbuffer.
   *
   * @param size Block size in elements.
   * @param rows Number of rows.
   * @param cols Number of cols.
   * @return matrix_cl wrapping the subbuffer.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  inline mat_t read_matrix_cl_(size_t size, int rows,
                                                       int cols) {
    align_pos();
    if (size == 0) {
      return mat_t(rows, cols);
    }
    check_r_capacity(size);
    const size_t origin_bytes = pos_r_ * sizeof(double);
    const size_t size_bytes = size * sizeof(double);
    cl_buffer_region region{origin_bytes, size_bytes};
    cl::Buffer parent = data_r_.buffer();
    cl::Buffer sub = parent.createSubBuffer(
        CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region);
    pos_r_ += size;
    align_pos();
    return mat_t(sub, rows, cols);
  }

 public:
  /**
   * Construct a deserializer over host buffers.
   *
   * @tparam RVec real data vector type.
   * @tparam IntVec integer data vector type.
   * @param data_r Real data.
   * @param data_i Integer data.
   */
  template <typename RVec, typename IntVec,
            require_all_vector_like_t<RVec, IntVec>* = nullptr>
  deserializer(const RVec& data_r, const IntVec& data_i)
      : data_r_(data_r),
        map_i_(data_i.data(), data_i.size()),
        r_size_(data_r.size()),
        i_size_(data_i.size()) {}

  /**
   * Construct a deserializer over an OpenCL buffer with alignment.
   *
   * @tparam IntVec integer data vector type.
   * @param data_r Device buffer of reals.
   * @param data_i Integer data.
   * @param align_elems Alignment in elements.
   */
  template <typename IntVec, require_vector_like_t<IntVec>* = nullptr>
  deserializer(const mat_t& data_r, const IntVec& data_i,
               size_t align_elems)
      : data_r_(data_r),
        map_i_(data_i.data(), data_i.size()),
        r_size_(data_r.size()),
        i_size_(data_i.size()),
        align_elems_(std::max<size_t>(1, align_elems)) {}

  /**
   * @return Number of remaining real elements.
   */
  inline size_t available() const noexcept { return r_size_ - pos_r_; }
  /**
   * @return Number of remaining integer elements.
   */
  inline size_t available_i() const noexcept { return i_size_ - pos_i_; }

  /**
   * Read a scalar floating-point value.
   *
   * @tparam Ret Floating point type.
   * @return Scalar value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret,
            require_t<std::is_floating_point<Ret>>* = nullptr>
  inline Ret read() {
    auto cl_val = read_matrix_cl_(1, 1, 1);
    return stan::math::from_matrix_cl<Ret>(cl_val);
  }

  /**
   * Read a complex scalar (two consecutive reals).
   *
   * @tparam Ret Complex type.
   * @return Complex value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, require_complex_t<Ret>* = nullptr>
  inline Ret read() {
    auto real = this->read<double>();
    auto imag = this->read<double>();
    return std::complex<double>{real, imag};
  }

  /**
   * Read an integer value.
   *
   * @tparam Ret Integral type.
   * @return Integer value.
   * @throws std::runtime_error if there are insufficient elements.
   */
  template <typename Ret, require_integral_t<Ret>* = nullptr>
  inline Ret read() {
    check_i_capacity(1);
    return map_i_.coeffRef(pos_i_++);
  }

  /**
   * Read a matrix_cl with the given dimensions.
   *
   * @tparam Ret matrix_cl type.
   * @param rows Rows.
   * @param cols Cols.
   * @return matrix_cl view of the subbuffer.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, require_matrix_cl_t<Ret>* = nullptr>
  inline Ret read(Eigen::Index rows, Eigen::Index cols) {
    return read_matrix_cl_(static_cast<size_t>(rows * cols),
                           static_cast<int>(rows), static_cast<int>(cols));
  }

  /**
   * Read a vector (matrix_cl with cols=1).
   *
   * @tparam Ret matrix_cl type.
   * @param m Length.
   * @return matrix_cl view of the subbuffer.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, require_matrix_cl_t<Ret>* = nullptr>
  inline Ret read(Eigen::Index m) {
    return read_matrix_cl_(static_cast<size_t>(m), static_cast<int>(m), 1);
  }

  /**
   * Read a std::vector of elements.
   *
   * @tparam Ret std::vector type.
   * @param m Vector length.
   * @param dims Dimensions for each element.
   * @return std::vector of deserialized elements.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read(Eigen::Index m, Sizes... dims) {
    std::decay_t<Ret> ret_vec;
    if (unlikely(m == 0)) {
      return ret_vec;
    }
    ret_vec.reserve(m);
    for (size_t i = 0; i < static_cast<size_t>(m); ++i) {
      ret_vec.emplace_back(this->read<value_type_t<Ret>>(dims...));
    }
    return ret_vec;
  }

  /**
   * Read with lower-bound constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam LB Lower bound type.
   * @tparam LP Log probability accumulator type.
   * @param lb Lower bound.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename LB, typename LP,
            typename... Sizes>
  inline auto read_constrain_lb(const LB& lb, LP& lp, Sizes... sizes) {
    return stan::math::lb_constrain<Jacobian>(this->read<Ret>(sizes...), lb, lp);
  }

  /**
   * Read with upper-bound constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam UB Upper bound type.
   * @tparam LP Log probability accumulator type.
   * @param ub Upper bound.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename UB, typename LP,
            typename... Sizes>
  inline auto read_constrain_ub(const UB& ub, LP& lp, Sizes... sizes) {
    return stan::math::ub_constrain<Jacobian>(this->read<Ret>(sizes...), ub, lp);
  }

  /**
   * Read with lower/upper-bound constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam LB Lower bound type.
   * @tparam UB Upper bound type.
   * @tparam LP Log probability accumulator type.
   * @param lb Lower bound.
   * @param ub Upper bound.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename LB, typename UB, typename LP,
            typename... Sizes>
  inline auto read_constrain_lub(const LB& lb, const UB& ub, LP& lp,
                                 Sizes... sizes) {
    return stan::math::lub_constrain<Jacobian>(this->read<Ret>(sizes...), lb, ub,
                                               lp);
  }

  /**
   * Read with offset-multiplier constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam M Offset type.
   * @tparam S Multiplier type.
   * @tparam LP Log probability accumulator type.
   * @param mu Offset.
   * @param sigma Multiplier.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename M, typename S, typename LP,
            typename... Sizes>
  inline auto read_constrain_offset_multiplier(const M& mu, const S& sigma,
                                               LP& lp, Sizes... sizes) {
    return stan::math::offset_multiplier_constrain<Jacobian>(
        this->read<Ret>(sizes...), mu, sigma, lp);
  }

  /**
   * Read with unit-vector constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam LP Log probability accumulator type.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes>
  inline auto read_constrain_unit_vector(LP& lp, Sizes... sizes) {
    return stan::math::unit_vector_constrain<Jacobian>(this->read<Ret>(sizes...),
                                                       lp);
  }
};

/**
 * OpenCL deserializer specialization for var_value<matrix_cl> buffers (rev).
 */
template <>
class deserializer<stan::math::var_value<stan::math::matrix_cl<double>>> {
 using mat_t = stan::math::matrix_cl<double>;
 private:
  std::reference_wrapper<mat_t> val_;
  std::reference_wrapper<mat_t> adj_;
  Eigen::Map<const Eigen::Matrix<int, -1, 1>> map_i_;
  size_t r_size_{0};
  size_t i_size_{0};
  size_t pos_r_{0};
  size_t pos_i_{0};
  size_t align_elems_{1};

  /**
   * Check there are at least m reals left to read.
   *
   * @param m Number of real elements to read.
   * @throws std::runtime_error if there are insufficient elements.
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
   * Check there are at least m integers left to read.
   *
   * @param m Number of integer elements to read.
   * @throws std::runtime_error if there are insufficient elements.
   */
  void check_i_capacity(size_t m) const {
    STAN_NO_RANGE_CHECKS_RETURN;
    if (pos_i_ + m > i_size_) {
      []() STAN_COLD_PATH {
        throw std::runtime_error("no more integers to read");
      }();
    }
  }

  /**
   * Align the real position to the next aligned element offset.
   */
  inline void align_pos() { pos_r_ = internal::round_up(pos_r_, align_elems_); }

  /**
   * Read a block as a var_value<matrix_cl> subbuffer.
   *
   * @param size Block size in elements.
   * @param rows Number of rows.
   * @param cols Number of cols.
   * @return var_value wrapping value and adjoint subbuffers.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  inline stan::math::var_value<mat_t>
  read_var_matrix_cl_(size_t size, int rows, int cols) {
    align_pos();
    if (size == 0) {
      mat_t empty_val(rows, cols);
      mat_t empty_adj(rows, cols);
      auto* vi = new stan::math::vari_value<mat_t>(
          std::move(empty_val), std::move(empty_adj));
      return stan::math::var_value<mat_t>(vi);
    }
    check_r_capacity(size);
    const size_t origin_bytes = pos_r_ * sizeof(double);
    const size_t size_bytes = size * sizeof(double);
    cl_buffer_region region{origin_bytes, size_bytes};
    cl::Buffer sub_val = val_.get().buffer().createSubBuffer(
        CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region);
    cl::Buffer sub_adj = adj_.get().buffer().createSubBuffer(
        CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region);
    pos_r_ += size;
    align_pos();
    mat_t val_mat(std::move(sub_val), rows, cols);
    mat_t adj_mat(std::move(sub_adj), rows, cols);
    auto* vi = new stan::math::vari_value<mat_t>(
        std::move(val_mat), std::move(adj_mat));
    return stan::math::var_value<mat_t>(vi);
  }

 public:
  /**
   * Construct a deserializer over a var_value<matrix_cl> buffer.
   *
   * @tparam IntVec integer data vector type.
   * @param data_r Device buffer with values and adjoints.
   * @param data_i Integer data.
   * @param align_elems Alignment in elements.
   */
  template <typename IntVec, require_vector_like_t<IntVec>* = nullptr>
  deserializer(const stan::math::var_value<mat_t>& data_r,
               const IntVec& data_i, size_t align_elems)
      : val_(data_r.val()),
        adj_(data_r.adj()),
        map_i_(data_i.data(), data_i.size()),
        r_size_(data_r.val().size()),
        i_size_(data_i.size()),
        align_elems_(std::max<size_t>(1, align_elems)) {}

  /**
   * @return Number of remaining real elements.
   */
  inline size_t available() const noexcept { return r_size_ - pos_r_; }
  /**
   * @return Number of remaining integer elements.
   */
  inline size_t available_i() const noexcept { return i_size_ - pos_i_; }

  /**
   * Read a scalar floating-point value.
   *
   * @tparam Ret Floating point type.
   * @return Scalar value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret,
            require_t<std::is_floating_point<Ret>>* = nullptr>
  inline Ret read() {
    auto cl_val = read_var_matrix_cl_(1, 1, 1).val();
    return stan::math::from_matrix_cl<Ret>(cl_val);
  }

  /**
   * Read a complex scalar (two consecutive reals).
   *
   * @tparam Ret Complex type.
   * @return Complex value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, require_complex_t<Ret>* = nullptr>
  inline Ret read() {
    auto real = this->read<double>();
    auto imag = this->read<double>();
    return std::complex<double>{real, imag};
  }

  /**
   * Read an integer value.
   *
   * @tparam Ret Integral type.
   * @return Integer value.
   * @throws std::runtime_error if there are insufficient elements.
   */
  template <typename Ret, require_integral_t<Ret>* = nullptr>
  inline Ret read() {
    check_i_capacity(1);
    return map_i_.coeffRef(pos_i_++);
  }

  /**
   * Read a var_value<matrix_cl> with the given dimensions.
   *
   * @tparam Ret var_value<matrix_cl> type.
   * @param rows Rows.
   * @param cols Cols.
   * @return var_value view of the subbuffers.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret,
            require_t<std::is_same<
                Ret,
                stan::math::var_value<mat_t>>>* = nullptr>
  inline Ret read(Eigen::Index rows, Eigen::Index cols) {
    return read_var_matrix_cl_(static_cast<size_t>(rows * cols),
                               static_cast<int>(rows),
                               static_cast<int>(cols));
  }

  /**
   * Read a vector as var_value<matrix_cl> (cols=1).
   *
   * @tparam Ret var_value<matrix_cl> type.
   * @param m Length.
   * @return var_value view of the subbuffers.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret,
            require_t<std::is_same<
                Ret,
                stan::math::var_value<mat_t>>>* = nullptr>
  inline Ret read(Eigen::Index m) {
    return read_var_matrix_cl_(static_cast<size_t>(m), static_cast<int>(m), 1);
  }

  /**
   * Read a std::vector of elements.
   *
   * @tparam Ret std::vector type.
   * @param m Vector length.
   * @param dims Dimensions for each element.
   * @return std::vector of deserialized elements.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, typename... Sizes,
            require_std_vector_t<Ret>* = nullptr>
  inline auto read(Eigen::Index m, Sizes... dims) {
    std::decay_t<Ret> ret_vec;
    if (unlikely(m == 0)) {
      return ret_vec;
    }
    ret_vec.reserve(m);
    for (size_t i = 0; i < static_cast<size_t>(m); ++i) {
      ret_vec.emplace_back(this->read<value_type_t<Ret>>(dims...));
    }
    return ret_vec;
  }

  /**
   * Read with lower-bound constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam LB Lower bound type.
   * @tparam LP Log probability accumulator type.
   * @param lb Lower bound.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename LB, typename LP,
            typename... Sizes>
  inline auto read_constrain_lb(const LB& lb, LP& lp, Sizes... sizes) {
    return stan::math::lb_constrain<Jacobian>(this->read<Ret>(sizes...), lb, lp);
  }

  /**
   * Read with upper-bound constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam UB Upper bound type.
   * @tparam LP Log probability accumulator type.
   * @param ub Upper bound.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename UB, typename LP,
            typename... Sizes>
  inline auto read_constrain_ub(const UB& ub, LP& lp, Sizes... sizes) {
    return stan::math::ub_constrain<Jacobian>(this->read<Ret>(sizes...), ub, lp);
  }

  /**
   * Read with lower/upper-bound constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam LB Lower bound type.
   * @tparam UB Upper bound type.
   * @tparam LP Log probability accumulator type.
   * @param lb Lower bound.
   * @param ub Upper bound.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename LB, typename UB, typename LP,
            typename... Sizes>
  inline auto read_constrain_lub(const LB& lb, const UB& ub, LP& lp,
                                 Sizes... sizes) {
    return stan::math::lub_constrain<Jacobian>(this->read<Ret>(sizes...), lb, ub,
                                               lp);
  }

  /**
   * Read with offset-multiplier constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam M Offset type.
   * @tparam S Multiplier type.
   * @tparam LP Log probability accumulator type.
   * @param mu Offset.
   * @param sigma Multiplier.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename M, typename S, typename LP,
            typename... Sizes>
  inline auto read_constrain_offset_multiplier(const M& mu, const S& sigma,
                                               LP& lp, Sizes... sizes) {
    return stan::math::offset_multiplier_constrain<Jacobian>(
        this->read<Ret>(sizes...), mu, sigma, lp);
  }

  /**
   * Read with unit-vector constraint.
   *
   * @tparam Ret Return type.
   * @tparam Jacobian Whether to include Jacobian.
   * @tparam LP Log probability accumulator type.
   * @param lp Log probability accumulator.
   * @param sizes Dimensions for the read.
   * @return Constrained value.
   * @throws std::runtime_error if there are insufficient elements.
   * @throws cl::Error if subbuffer creation fails.
   */
  template <typename Ret, bool Jacobian, typename LP, typename... Sizes>
  inline auto read_constrain_unit_vector(LP& lp, Sizes... sizes) {
    return stan::math::unit_vector_constrain<Jacobian>(this->read<Ret>(sizes...),
                                                       lp);
  }
};

}  // namespace io
}  // namespace stan

#endif  // STAN_OPENCL

#endif  // STAN_IO_OPENCL_DESERIALIZER_HPP
