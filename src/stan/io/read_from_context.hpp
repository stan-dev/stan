#ifndef STAN_IO_READ_FROM_CONTEXT_HPP
#define STAN_IO_READ_FROM_CONTEXT_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/num_elements.hpp>
#include <stan/math/prim/meta/is_complex.hpp>
#include <stan/math/prim/meta/is_tuple.hpp>
#include <stan/math/prim/meta/index_apply.hpp>
#include <stan/math/prim/meta/scalar_type.hpp>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace stan {

namespace internal {

/**
 * @brief Read one integer without advancing a consumption cursor.
 * @param[out] x Destination integer.
 * @param[in] values Buffer of integer values for a named leaf.
 * @param[in] idx Zero-based index of the value to read.
 * @throw std::runtime_error if idx is outside values.
 */
inline void read_from_buffer(int& x, const std::vector<int>& values,
                             std::size_t idx) {
  if (idx >= values.size()) {
    throw std::runtime_error("read_from_context: not enough integer values");
  }
  x = values[idx];
}

/**
 * @brief Read one real value without advancing a consumption cursor.
 * @param[out] x Destination real value.
 * @param[in] values Buffer of real values for a named leaf.
 * @param[in] idx Zero-based index of the value to read.
 * @throw std::runtime_error if idx is outside values.
 */
inline void read_from_buffer(double& x, const std::vector<double>& values,
                             std::size_t idx) {
  if (idx >= values.size()) {
    throw std::runtime_error("read_from_context: not enough real values");
  }
  x = values[idx];
}

/**
 * @brief Read one complex value from separate component indices.
 * @param[out] x Destination complex value.
 * @param[in] values Buffer of real and imaginary components for a named leaf.
 * @param[in] real_idx Zero-based index of the real component.
 * @param[in] imag_idx Zero-based index of the imaginary component.
 * @throw std::runtime_error if either index is outside values.
 */
inline void read_from_buffer(std::complex<double>& x,
                             const std::vector<double>& values,
                             std::size_t real_idx, std::size_t imag_idx) {
  if (real_idx >= values.size() || imag_idx >= values.size()) {
    throw std::runtime_error(
        "read_from_context: not enough complex components");
  }
  x = {values[real_idx], values[imag_idx]};
}

/**
 * @brief Read an Eigen object in logical column-major order without advancing
 * a consumption cursor.
 * @tparam EigMat Writable Eigen destination type with int, double, or
 * std::complex<double> coefficients.
 * @tparam InVec Contiguous input buffer type with int or double coefficients,
 * possibly const. Integer destinations use integer buffers; real and complex
 * destinations use real buffers.
 * @param[in,out] x Destination with its rows and columns already allocated.
 * @param[in] values Buffer containing the tuple-free payload to read.
 * @param[in] offset Zero-based source index of the first coefficient's value
 * or real component.
 * @param[in] stride Positive distance in buffer elements between successive
 * logical column-major coefficients.
 * @param[in] imaginary_offset Distance in buffer elements from each real
 * component to its imaginary component; ignored for non-complex destinations.
 * @pre The caller has validated that all source indices are within values.
 */
template <typename EigMat, typename InVec, require_eigen_t<EigMat>* = nullptr>
void read_from_buffer(EigMat& x, InVec& values, std::size_t offset,
                      std::size_t stride, std::size_t imaginary_offset) {
  using Scalar = scalar_type_t<EigMat>;
  const std::size_t size = x.size();
  if (size == 0) {
    return;
  }
  using Source
      = Eigen::Matrix<scalar_type_t<InVec>, Eigen::Dynamic, Eigen::Dynamic>;
  using Stride = Eigen::InnerStride<Eigen::Dynamic>;
  using SourceMap = Eigen::Map<const Source, Eigen::Unaligned, Stride>;
  if constexpr (stan::is_complex<Scalar>::value) {
    x.real()
        = SourceMap(values.data() + offset, x.rows(), x.cols(), Stride(stride));
    x.imag() = SourceMap(values.data() + offset + imaginary_offset, x.rows(),
                         x.cols(), Stride(stride));
  } else {
    x = SourceMap(values.data() + offset, x.rows(), x.cols(), Stride(stride));
  }
}

/**
 * @brief Read a tuple-free array using column-major source strides.
 * Each child uses stride * x.size(); imaginary_offset stays constant within
 * the payload. This function does not advance a consumption cursor.
 * @tparam StdVec Destination std::vector type containing int, double,
 * std::complex<double>, Eigen objects, or nested tuple-free std::vectors.
 * @tparam InVec Input std::vector type holding int values for integer
 * destinations or double values for real and complex destinations.
 * @param[in,out] x Rectangular destination with every dimension allocated.
 * @param[in] values Buffer containing the tuple-free payload to read.
 * @param[in] offset Zero-based source index of the first value or real
 * component in this array.
 * @param[in] stride Positive distance in buffer elements between the starting
 * indices of consecutive elements of x.
 * @param[in] imaginary_offset Distance in buffer elements from each real
 * component to its imaginary component; ignored for non-complex destinations.
 * @pre The caller has validated that all source indices are within values.
 */
template <typename StdVec, typename InVec,
          require_std_vector_t<StdVec>* = nullptr>
void read_from_buffer(StdVec& x, const InVec& values, std::size_t offset,
                      std::size_t stride, std::size_t imaginary_offset) {
  using T = value_type_t<StdVec>;
  if constexpr (stan::is_complex<T>::value) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      const std::size_t idx = offset + i * stride;
      internal::read_from_buffer(x[i], values, idx, idx + imaginary_offset);
    }
  } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, int>) {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Stride = Eigen::InnerStride<Eigen::Dynamic>;
    Eigen::Map<Vec>(x.data(), x.size())
        = Eigen::Map<const Vec, Eigen::Unaligned, Stride>(
            values.data() + offset, x.size(), Stride(stride));
  } else {
    for (std::size_t i = 0; i < x.size(); ++i) {
      const std::size_t idx = offset + i * stride;
      internal::read_from_buffer(x[i], values, idx, stride * x.size(),
                                 imaginary_offset);
    }
  }
}

/**
 * @brief Read one tuple-free payload and advance its consumption cursor.
 * Validate the full payload before reading. Complex payloads use separate
 * real and imaginary component blocks and consume two stored values per
 * destination value. Non-complex payloads consume one stored value per
 * destination value.
 * @tparam T Tuple-free destination type: int, double, std::complex<double>,
 * an Eigen object, or a rectangular std::vector nesting of these types.
 * @tparam InputVec Input std::vector type, or a reference to it, holding int
 * values for integer destinations or double values for real and complex ones.
 * @param[in,out] x Destination with every container dimension allocated.
 * @param[in] values Buffer for the named leaf containing this payload;
 * neither modified nor moved from.
 * @param[in,out] position Zero-based start of the payload in values. Advances
 * by the number of stored int or double values on success, counting two per
 * complex value. An empty payload leaves position unchanged.
 * @throw std::runtime_error if position exceeds the buffer size or there are
 * insufficient values for the payload.
 */
template <typename T, typename InputVec>
void read_from_buffer_block(T& x, InputVec&& values, std::size_t& position) {
  const std::size_t size = math::num_elements(x);
  if (position > values.size()) {
    throw std::runtime_error(
        "read_from_context: not enough values for payload");
  }
  if (size == 0) {
    return;
  }
  if constexpr (stan::is_complex<scalar_type_t<T>>::value) {
    if (size > (values.size() - position) / 2) {
      throw std::runtime_error(
          "read_from_context: not enough values for complex payload");
    }
    if constexpr (stan::is_complex<T>::value) {
      internal::read_from_buffer(x, values, position, position + 1);
    } else {
      internal::read_from_buffer(x, values, position, 1, size);
    }
    position += 2 * size;
  } else {
    if (size > values.size() - position) {
      throw std::runtime_error(
          "read_from_context: not enough values for payload");
    }
    if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
      internal::read_from_buffer(x, values, position);
    } else {
      internal::read_from_buffer(x, values, position, 1, 0);
    }
    position += size;
  }
}

/**
 * @brief Read a complete tuple-free payload once the field path is exhausted.
 * @tparam T Tuple-free scalar, Eigen, or rectangular std::vector destination
 * type with int, double, or std::complex<double> scalar values.
 * @tparam InVec Input std::vector type holding int values for integer
 * destinations or double values for real and complex destinations.
 * @param[in,out] x Destination payload with its dimensions already allocated.
 * @param[in] values Buffer for the named leaf containing this payload.
 * @param[in,out] position Zero-based payload start in values; advances past
 * this payload on success, counting both components of complex values.
 * @param[in] fields Empty tuple-field path selecting x itself as the payload.
 * @throw std::runtime_error if the buffer cannot supply the complete payload.
 */
template <typename T, typename InVec>
void read_from_buffer(T& x, const InVec& values, std::size_t& position,
                      std::index_sequence<> fields) {
  internal::read_from_buffer_block(x, values, position);
}

/**
 * @brief Follow a tuple-field path through the destination for one named leaf.
 * Tuples consume one path index. Arrays visit elements in order without
 * consuming an index, sharing the input buffer and consumption cursor.
 * @tparam T Destination tuple or std::vector nesting containing tuples.
 * @tparam InVec Input std::vector type holding int values for integer
 * destinations or double values for real and complex destinations.
 * @tparam First Zero-based field index in the next tuple encountered.
 * @tparam Rest Remaining zero-based field indices in successive nested tuples.
 * @param[in,out] x Destination subtree with every dimension already allocated.
 * @param[in] values Buffer for the named leaf selected by fields.
 * @param[in,out] position Zero-based index of the next unread buffer element;
 * advances through each selected payload in traversal order.
 * @param[in] fields Compile-time tuple-field path, excluding array indices.
 * @throw std::runtime_error if the buffer cannot supply a selected payload.
 */
template <typename T, typename InVec, std::size_t First, std::size_t... Rest>
void read_from_buffer(T& x, const InVec& values, std::size_t& position,
                      std::index_sequence<First, Rest...> fields) {
  if constexpr (is_tuple_v<T>) {
    internal::read_from_buffer(std::get<First>(x), values, position,
                               std::index_sequence<Rest...>{});
  } else {
    for (auto& element : x) {
      internal::read_from_buffer(element, values, position, fields);
    }
  }
}

/**
 * @brief Check that a named leaf buffer has been consumed exactly.
 * @tparam InVec Input buffer type providing size().
 * @param[in] values Loaded buffer for one named leaf.
 * @param[in] position Number of consumed buffer elements, counting both
 * components of complex values.
 * @param[in] path Variable name, including one-based dotted tuple-field
 * suffixes, used in the error message.
 * @throw std::runtime_error if position differs from values.size().
 */
template <typename InVec>
void check_consumed(const InVec& values, std::size_t position,
                    const std::string& path) {
  if (position != values.size()) {
    throw std::runtime_error("read_from_context: unexpected value count for "
                             + path);
  }
}

/**
 * @brief Load and write one named leaf at a time, releasing each buffer before
 * loading the next. Discover tuple paths from types, including for empty
 * arrays.
 * @tparam Field Type at the current tuple-field path, possibly wrapped in
 * arrays. Its scalar_type_t determines the next tuple or the leaf scalar type.
 * @tparam Root Complete destination type with int, double, or
 * std::complex<double> scalar values, Eigen objects, std::vectors, and tuples.
 * @tparam Context Source type providing const vals_i(name) and vals_r(name)
 * methods returning std::vector<int> and std::vector<double>, respectively.
 * @tparam Fields Zero-based tuple-field indices from the root, excluding array
 * indices.
 * @param[in,out] x Complete rectangular destination with all dimensions
 * allocated. Earlier writes remain if a subsequent leaf fails.
 * @param[in] context Source of the named integer and real buffers.
 * @param[in] path Variable name including the one-based dotted tuple-field
 * suffixes corresponding to Fields.
 * @param[in] fields Compile-time tuple-field path into x corresponding to
 * Field and path; empty at the root.
 * @throw std::runtime_error if a leaf has too few or too many values for its
 * destinations. Exceptions from the context propagate to the caller.
 */
template <typename Field, typename Root, typename Context,
          std::size_t... Fields>
void load_from_context(Root& x, const Context& context, const std::string& path,
                       std::index_sequence<Fields...> fields) {
  using S = scalar_type_t<Field>;
  if constexpr (is_tuple_v<S>) {
    math::index_apply<std::tuple_size_v<S>>([&x, &context, &path](auto... i) {
      (internal::load_from_context<std::tuple_element_t<decltype(i)::value, S>>(
           x, context, path + "." + std::to_string(decltype(i)::value + 1),
           std::index_sequence<Fields..., decltype(i)::value>{}),
       ...);
    });
  } else {
    std::size_t position = 0;
    if constexpr (std::is_same_v<S, int>) {
      const auto values = context.vals_i(path);
      internal::read_from_buffer(x, values, position, fields);
      internal::check_consumed(values, position, path);
    } else {
      static_assert(
          std::is_same_v<S, double> || std::is_same_v<S, std::complex<double>>,
          "read_from_context requires int, double or complex<double> "
          "scalar types");
      const auto values = context.vals_r(path);
      internal::read_from_buffer(x, values, position, fields);
      internal::check_consumed(values, position, path);
    }
  }
}

}  // namespace internal

/**
 * Read a named variable into an already-sized destination.
 *
 * Supported destinations are int, double, std::complex<double>, Eigen
 * vectors, row vectors and matrices, and nested std::vectors and std::tuples
 * of these types. Arrays must be rectangular. Declared dimensions must be
 * validated before this call; empty containers cannot describe the sizes of
 * their missing elements.
 *
 * Each dotted tuple-field name is fetched once, written, and its buffer
 * released before the next name is fetched. Arrays containing tuples are
 * traversed in element order, while each tuple-free payload is decoded in
 * column-major order. Complex components are paired within that payload.
 * If a later leaf fails, previously written leaves remain modified.
 *
 * @tparam T Destination type: int, double, std::complex<double>, an Eigen
 * object, or a nesting of std::vectors and std::tuples of supported types.
 * @tparam Context Source type providing const vals_i(name) and vals_r(name)
 * methods returning std::vector<int> and std::vector<double>, respectively.
 * @param[in,out] x Rectangular destination with its declared dimensions already
 * allocated. Scalar values are overwritten; container dimensions are preserved.
 * @param[in] context Source of the named integer and real buffers.
 * @param[in] name Root variable name, without array indices or tuple suffixes;
 * converted to an owned string for this call.
 * @throw std::runtime_error if the input contains an unexpected number of
 * values.
 */
template <typename T, typename Context>
inline void read_from_context(T& x, const Context& context,
                              std::string_view name) {
  const std::string path(name);
  if constexpr (stan::is_complex<T>::value) {
    static_assert(std::is_same_v<typename T::value_type, double>,
                  "read_from_context requires complex<double> scalar types");
    const auto values = context.vals_r(path);
    if (values.size() != 2) {
      throw std::runtime_error("read_from_context: expected two components for "
                               + path);
    }
    x = {values[0], values[1]};
  } else if constexpr (std::is_same_v<T, int>) {
    const auto values = context.vals_i(path);
    if (values.size() != 1) {
      throw std::runtime_error("read_from_context: expected one value for "
                               + path);
    }
    x = values[0];
  } else if constexpr (std::is_same_v<T, double>) {
    const auto values = context.vals_r(path);
    if (values.size() != 1) {
      throw std::runtime_error("read_from_context: expected one value for "
                               + path);
    }
    x = values[0];
  } else {
    internal::load_from_context<T>(x, context, path, std::index_sequence<>{});
  }
}

}  // namespace stan

#endif
