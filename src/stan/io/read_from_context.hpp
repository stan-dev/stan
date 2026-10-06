#ifndef STAN_IO_READ_FROM_CONTEXT_HPP
#define STAN_IO_READ_FROM_CONTEXT_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/num_elements.hpp>
#include <stan/math/prim/meta/contains_tuple.hpp>
#include <stan/math/prim/meta/index_apply.hpp>
#include <stan/math/prim/meta/is_complex.hpp>
#include <stan/math/prim/meta/is_eigen.hpp>
#include <stan/math/prim/meta/is_tuple.hpp>
#include <stan/math/prim/meta/is_vector.hpp>
#include <stan/math/prim/meta/scalar_type.hpp>
#include <stan/math/prim/meta/value_type.hpp>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace stan {

namespace io {

/* A var_context stores one value array per NAME, never per array element. For
 *
 *   array[2] tuple(real, real) x;
 *
 * it holds exactly two entries, "x.1" -> {a, b} and "x.2" -> {c, d}. There is
 * no "x[0].1". Reading is therefore name-major: fetch a name once, then visit
 * every element that has that slot. Element-major would refetch each buffer
 * once per element, and vals_r returns its vector by value.
 *
 * Worked example:
 *
 *   using Leaf = std::tuple<Eigen::MatrixXcd, double>;
 *   using Mid = std::tuple<std::vector<Leaf>>;
 *   std::vector<Mid> x;
 *
 * scalar_type_t erases every std::vector and Eigen layer and keeps the tuples,
 * so SlotT starts as the name tree alone:
 *
 *   SlotT = std::tuple<std::tuple<std::complex<double>, double>>
 *
 * read_from_context walks that tree at compile time and never touches x. The
 * std::tuple_element_t below indexes a TYPE; the only runtime work in that
 * branch is building name strings.
 *
 *   SlotT                        path     name
 *   tuple<tuple<complex, double>>   <>       "x"
 *     \__ tuple<complex, double>    <0>      "x.1"
 *          |__ complex<double>      <0,0>    "x.1.1"   leaf
 *          \__ double               <0,1>    "x.1.2"   leaf
 *
 * Each leaf fetches its own buffer, then fill_slots walks the VALUE to every
 * destination that name feeds, off one shared cursor. For "x.1.1", path <0,0>:
 *
 *   x             std::vector<Mid>    array  -> loop, path stays <0,0>
 *    \__ x[i]     std::tuple          slot 0 -> get<0>, path <0>
 *         \__ v   std::vector<Leaf>   array  -> loop, path stays <0>
 *              \__ v[j]  std::tuple   slot 0, path empty -> the payload
 *
 * One buffer fills the matrix of every x[i] -> v[j], laid back to back.
 *
 * The two array orderings are opposite, and both are deliberate:
 *   an array ENCLOSING a tuple is element order        (fill_slots)
 *   a tuple-free payload is column-major               (read_payload)
 * array_array_real and nested_array_tuple are both 2x3 and disagree.
 */

namespace internal {

template <typename ScalarType, typename ContextT, typename Name>
inline auto get_values(const ContextT& context, const Name& name) {
  if constexpr (std::is_same_v<ScalarType, int>) {
    return context.vals_i(name);
  } else {
    return context.vals_r(name);
  }
}

template <typename InVec>
using source_map_t = Eigen::Map<
    const Eigen::Matrix<value_type_t<InVec>, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::Unaligned, Eigen::InnerStride<Eigen::Dynamic>>;

template <typename T, typename InVec>
inline void read_payload(T& x, const InVec& values, std::size_t offset,
                         std::size_t stride) {
  using map_t = source_map_t<InVec>;
  if constexpr (std::is_arithmetic_v<T>) {
    x = values[offset];
  } else if constexpr (is_eigen_v<T>) {
    x = map_t(values.data() + offset, x.rows(), x.cols(), stride);
  } else if constexpr (std::is_same_v<value_type_t<T>, scalar_type_t<T>>) {
    const std::size_t size = x.size();
    Eigen::Map<Eigen::Matrix<scalar_type_t<T>, Eigen::Dynamic, 1>> dst(x.data(),
                                                                       size);
    dst = map_t(values.data() + offset, size, 1, stride);
  } else {
    const std::size_t size = x.size();
    const std::size_t child_stride = stride * size;
    for (std::size_t i = 0; i < size; ++i) {
      read_payload(x[i], values, offset + i * stride, child_stride);
    }
  }
}

template <typename T, typename InVec>
inline void read_payload(T& x, const InVec& values, std::size_t offset) {
  using map_t = source_map_t<InVec>;
  if constexpr (std::is_arithmetic_v<T>) {
    x = values[offset];
  } else if constexpr (is_eigen_v<T>) {
    x = map_t(values.data() + offset, x.rows(), x.cols(), 1);
  } else if constexpr (std::is_same_v<value_type_t<T>, scalar_type_t<T>>) {
    const std::size_t size = x.size();
    Eigen::Map<Eigen::Matrix<scalar_type_t<T>, Eigen::Dynamic, 1>> dst(x.data(),
                                                                       size);
    dst = map_t(values.data() + offset, size, 1, 1);
  } else {
    const std::size_t size = x.size();
    const std::size_t child_stride = size;
    for (std::size_t i = 0; i < size; ++i) {
      read_payload(x[i], values, offset + i, child_stride);
    }
  }
}

template <typename T, typename InVec>
inline void read_payload(T& x, const InVec& values) {
  using map_t = source_map_t<InVec>;
  if constexpr (std::is_arithmetic_v<T>) {
    x = values[0];
  } else if constexpr (is_eigen_v<T>) {
    x = map_t(values.data(), x.rows(), x.cols(), 1);
  } else if constexpr (std::is_same_v<value_type_t<T>, scalar_type_t<T>>) {
    const std::size_t size = x.size();
    Eigen::Map<Eigen::Matrix<scalar_type_t<T>, Eigen::Dynamic, 1>> dst(x.data(),
                                                                       size);
    dst = map_t(values.data(), size, 1, 1);
  } else {
    const std::size_t size = x.size();
    const std::size_t child_stride = size;
    for (std::size_t i = 0; i < size; ++i) {
      read_payload(x[i], values, i, child_stride);
    }
  }
}

template <typename T, typename InVec>
inline void read_payload_complex(T& x, const InVec& values, std::size_t offset,
                                 std::size_t stride,
                                 std::size_t imaginary_offset) {
  using map_t = source_map_t<InVec>;
  if constexpr (is_complex<T>::value) {
    x = {values[offset], values[offset + imaginary_offset]};
  } else if constexpr (is_eigen_v<T>) {
    x.real() = map_t(values.data() + offset, x.rows(), x.cols(), stride);
    x.imag() = map_t(values.data() + offset + imaginary_offset, x.rows(),
                     x.cols(), stride);
  } else if constexpr (std::is_same_v<value_type_t<T>, scalar_type_t<T>>) {
    const std::size_t size = x.size();
    Eigen::Map<Eigen::Matrix<scalar_type_t<T>, Eigen::Dynamic, 1>> dst(x.data(),
                                                                       size);
    dst.real() = map_t(values.data() + offset, size, 1, stride);
    dst.imag()
        = map_t(values.data() + offset + imaginary_offset, size, 1, stride);
  } else {
    const std::size_t size = x.size();
    const std::size_t child_stride = stride * size;
    for (std::size_t i = 0; i < size; ++i) {
      read_payload_complex(x[i], values, offset + i * stride, child_stride,
                           imaginary_offset);
    }
  }
}

template <typename T, typename InVec>
inline void read_payload_complex(T& x, const InVec& values,
                                 std::size_t imaginary_offset) {
  using map_t = source_map_t<InVec>;
  if constexpr (is_complex<T>::value) {
    x = {values[0], values[imaginary_offset]};
  } else if constexpr (is_eigen_v<T>) {
    x.real() = map_t(values.data(), x.rows(), x.cols(), 1);
    x.imag() = map_t(values.data() + imaginary_offset, x.rows(), x.cols(), 1);
  } else if constexpr (std::is_same_v<value_type_t<T>, scalar_type_t<T>>) {
    const std::size_t size = x.size();
    Eigen::Map<Eigen::Matrix<scalar_type_t<T>, Eigen::Dynamic, 1>> dst(x.data(),
                                                                       size);
    dst.real() = map_t(values.data() + 0, size, 1, 1);
    dst.imag() = map_t(values.data() + 0 + imaginary_offset, size, 1, 1);
  } else {
    const std::size_t size = x.size();
    const std::size_t child_stride = size;
    for (std::size_t i = 0; i < size; ++i) {
      read_payload_complex(x[i], values, i, child_stride, imaginary_offset);
    }
  }
}

template <typename T, typename InVec, std::size_t Slot, std::size_t... Rest>
inline void fill_slots(T& x, const InVec& values, std::size_t& cursor,
                       std::index_sequence<Slot, Rest...> path) {
  if constexpr (!is_tuple_v<T>) {
    for (auto& element : x) {
      fill_slots(element, values, cursor, path);
    }
  } else if constexpr (sizeof...(Rest) > 0) {
    fill_slots(std::get<Slot>(x), values, cursor,
               std::index_sequence<Rest...>{});
  } else {
    using scalar_t = scalar_type_t<std::tuple_element_t<Slot, T>>;
    auto& payload = std::get<Slot>(x);
    const std::size_t size = math::num_elements(payload);
    const std::size_t needed = is_complex<scalar_t>::value ? 2 * size : size;
    if (needed > values.size() - cursor) {
      throw std::runtime_error(
          "read_from_context: ran out of values filling "
          "slot "
          + std::to_string(Slot + 1));
    } else if (size > 0) {
      if constexpr (is_complex<scalar_t>::value) {
        read_payload_complex(payload, values, cursor, 1, size);
      } else {
        read_payload(payload, values, cursor);
      }
    }
    cursor += needed;
  }
}

}  // namespace internal

/**
 * Read a tuple-free variable out of a var_context into an already-sized
 * destination.
 *
 * The whole destination is one name and one whole buffer, so the buffer is
 * fetched once and must supply exactly the number of values the destination
 * holds. A complex destination takes two values per coefficient, every real
 * component before every imaginary one.
 *
 * @tparam T Destination type with no std::tuple anywhere inside: int, double,
 * std::complex<double>, an Eigen vector, row vector or matrix, or a
 * rectangular std::vector nesting of these. Its scalar type must be int,
 * double or std::complex<double>.
 * @tparam Context Source providing vals_i(name) and vals_r(name) as const
 * members returning std::vector<int> and std::vector<double> by value.
 * Integer destinations read vals_i, real and complex ones read vals_r.
 * @param[in,out] x Destination with every dimension already allocated.
 * Declared dimensions must be validated before this call, because an empty
 * container cannot describe the sizes of the elements it does not have.
 * @param[in] context Source of the named integer and real buffers.
 * @param[in] name Variable name, with no array indices or tuple suffixes.
 * @throw std::runtime_error if the context supplies a number of values other
 * than the number the destination holds. Exceptions from the context
 * propagate unchanged.
 */
template <typename T, typename Context,
          stan::require_not_t<stan::contains_tuple<T>>* = nullptr>
inline void read_from_context(T& x, const Context& context,
                              const std::string& name) {
  using scalar_t = scalar_type_t<T>;
  static_assert(
      std::is_arithmetic_v<scalar_t> || stan::is_complex<scalar_t>::value,
      "read_from_context requires int, double or complex<double> "
      "scalars");
  const auto values = internal::get_values<scalar_t>(context, name);
  const std::size_t size = math::num_elements(x);
  const std::size_t expected = is_complex<scalar_t>::value ? 2 * size : size;
  if (values.size() != expected) {
    throw std::runtime_error("read_from_context: " + name + " expected "
                             + std::to_string(expected) + " values, got "
                             + std::to_string(values.size()));
  } else if (size == 0) {
    return;
  }
  if constexpr (std::is_arithmetic_v<T>) {
    x = values[0];
  } else if constexpr (is_complex<T>::value) {
    x = {values[0], values[1]};
  } else if constexpr (is_eigen_v<T>) {
    using map_t
        = Eigen::Map<const Eigen::Matrix<value_type_t<decltype(values)>,
                                         Eigen::Dynamic, Eigen::Dynamic>>;
    if constexpr (is_complex<scalar_t>::value) {
      x.real() = map_t(values.data(), x.rows(), x.cols());
      x.imag() = map_t(values.data() + size, x.rows(), x.cols());
    } else {
      x = map_t(values.data(), x.rows(), x.cols());
    }
  } else if constexpr (std::is_same_v<value_type_t<T>, scalar_t>) {
    if constexpr (is_complex<scalar_t>::value) {
      for (std::size_t i = 0; i < size; ++i) {
        x[i] = {values[i], values[i + size]};
      }
    } else {
      std::copy(values.begin(), values.end(), x.begin());
    }
  } else if constexpr (is_complex<scalar_t>::value) {
    internal::read_payload_complex(x, values, size);
  } else {
    internal::read_payload(x, values);
  }
}

/**
 * Read a variable containing std::tuple out of a var_context into an
 * already-sized destination.
 *
 * Every tuple slot is a separate name, so this reads one name per leaf of the
 * tuple skeleton, fetching and releasing each buffer before the next. If a
 * later name fails, the names already read remain written. See the file
 * comment above for a worked example of the two traversals.
 *
 * Call it with three arguments; SlotT and path carry the recursion and
 * default to the root of the tuple skeleton and the empty path.
 *
 * @tparam T Destination type with a std::tuple somewhere inside: a
 * std::tuple, or a rectangular std::vector nesting around one. Every leaf
 * scalar must be int, double or std::complex<double>.
 * @tparam SlotT Position in the tuple skeleton, scalar_type_t<T> at the
 * root. A std::tuple here means the name has more slots below it; anything
 * else means the name is a leaf and is read. Never a type of x itself, since
 * scalar_type_t has erased the std::vector and Eigen layers.
 * @tparam Context Source providing vals_i(name) and vals_r(name) as const
 * members returning std::vector<int> and std::vector<double> by value.
 * @tparam Slots Tuple slot indices from the root to SlotT, excluding array
 * indices. The path that fill_slots follows through the destination.
 * @param[in,out] x Destination with every dimension already allocated. It is
 * always the root: the recursion descends SlotT and path, never x.
 * @param[in] context Source of the named integer and real buffers.
 * @param[in] name Variable name, with the dotted one-based suffix for Slots
 * already appended.
 * @param[in] path Slots as a value, so the pack is deduced rather than given.
 * @throw std::runtime_error if a name supplies a number of values other than
 * the number its destinations hold. Exceptions from the context propagate
 * unchanged.
 */
template <typename T, typename SlotT = scalar_type_t<T>, typename Context,
          std::size_t... Slots,
          stan::require_t<stan::contains_tuple<T>>* = nullptr>
inline void read_from_context(T& x, const Context& context,
                              std::string_view name,
                              std::index_sequence<Slots...> path = {}) {
  if constexpr (is_tuple_v<SlotT>) {
    math::index_apply<std::tuple_size_v<SlotT>>([&x, &context,
                                                 &name](auto... Slot) {
      (read_from_context<T, std::tuple_element_t<Slot.value, SlotT>>(
           x, context, std::string(name) + "." + std::to_string(Slot.value + 1),
           std::index_sequence<Slots..., Slot.value>{}),
       ...);
    });
  } else {
    constexpr bool is_valid_slot_type
        = std::is_arithmetic_v<SlotT> || stan::is_complex<SlotT>::value;
    static_assert(is_valid_slot_type,
                  "read_from_context requires int, double or complex<double> "
                  "scalars");
    const std::string leaf_name(name);
    const auto values = internal::get_values<SlotT>(context, leaf_name);
    std::size_t cursor = 0;
    internal::fill_slots(x, values, cursor, path);
    if (cursor != values.size()) {
      throw std::runtime_error("read_from_context: " + leaf_name + " supplied "
                               + std::to_string(values.size())
                               + " values but the destination holds "
                               + std::to_string(cursor));
    }
  }
}

}  // namespace io

}  // namespace stan

#endif
