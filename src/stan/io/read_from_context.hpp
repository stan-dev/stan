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
#include <stan/math/prim/meta/require_helpers.hpp>
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

/** 
 * `read_from_context` is used by the stan compiler to read in user data
 *  from a `var_context`. This code looks a bit messy due to the indexing
 *  issues that arise from serializing arrays of tuples and complex number 
 *  types.  
 * `read_from_context` fills one already-sized model variable from a
 * `var_context`. The context maps a name to a flat `std::vector<int>` or
 * `std::vector<double>`, while the destination can be any nesting of
 * `std::vector`, Eigen types and `std::tuple`. The stan io model has three
 * non-trivial rules for arrays of tuples and complex numbers that make
 * the io challenging.
 *
 * 1. Arrays are column-major, first index fastest. For `array[2] vector[3] v`
 *    the buffer is `v[0][0], v[1][0], v[0][1], v[1][1], ...`, so `v[0]` is
 *    every second value rather than a contiguous run. `read_payload` tracks
 *    this with an `offset` and a `stride` that is multiplied by each array
 *    length it descends through.
 *
 * 2. Complex values are stored as every real component followed by every
 *    imaginary one. A complex payload of `n` coefficients takes `2n` values,
 *    and each imaginary part sits `n` after its real part.
 *    `read_payload_complex` carries that `n` as `imaginary_offset`.
 *
 * 3. A tuple is not one entry. Each tuple slot is its own name (`"x.1"`,
 *    `"x.2.1"`), and array indices never appear in a name. For
 *
 * ```stan
 * array[3] tuple(real, real) x;
 * ```
 *
 *    the context holds only `"x.1"` and `"x.2"`, with 3 values each. One name
 *    feeds that slot in every element of the enclosing array, and each
 *    element's share is a contiguous block in element order. This is the
 *    opposite of rule 1: an array enclosing a tuple is element order, while an
 *    array inside a tuple slot is column-major.
 *
 * Rule 3 sets the structure of the file. A destination with no tuple is one
 * name and one buffer, so the first overload reads it directly. A destination
 * containing a tuple is read one name at a time. The second overload walks the
 * tuple types to build each leaf name and fetches that name's buffer once.
 * `fill_slots` then walks the destination to every place the name feeds,
 * taking one block per place off a shared `cursor`. Walking elements first
 * would instead fetch every buffer once per element, and `vals_r` returns by
 * value.
 *
 * Example:
 * Our example will go over the array of tuples with nested arrays and vectors.
 * ```stan
 * array[2] tuple(int, array[2] vector[2]) x;
 * ```
 * This is the data structure below with nested arrays in a tuple.
 * ```cpp
 * std::vector<std::tuple<int, std::vector<Eigen::VectorXd>>> x;
 * ```
 *
 * The input JSON will look like the following:
 *
 * ```
 * "x": [{"1": 7, "2": [[1, 2], [3, 4]]},
 *       {"1": 8, "2": [[5, 6], [7, 8]]}]
 * ```
 *
 * The context holds two names, one per tuple slot:
 *
 * ```
 * "x.1"  {7, 8}                          int
 *         |  \__ x[2].1
 *         \_____ x[1].1
 *
 * "x.2"  {1, 3, 2, 4,  5, 7, 6, 8}       array[2] vector[2]
 *         \________/  \________/
 *           x[1].2      x[2].2           element order (rule 3)
 *
 * Inside the x[1].2 block, first index fastest (rule 1):
 *
 *   1 -> x[1].2[1][1]    
 *   2 -> x[1].2[1][2]
 *   3 -> x[1].2[2][1]    
 *   4 -> x[1].2[2][2]
 *
 * so x[1].2[1] = [1, 2]' and x[1].2[2] = [3, 4]'. x[2].2 is read the
 * same way from {5, 7, 6, 8}.
 * ```
 *  
 * `scalar_type_t<decltype(x)>` is `std::tuple<int, double>`, so the second
 * overload reads `"x.1"` through `vals_i` and `"x.2"` through `vals_r`. For
 * `"x.2"`, `fill_slots` visits `x[0]` and then `x[1]`, takes slot 2 of each,
 * and hands `read_payload` the next block of 4 values. The slot's array has
 * length 2, so the stride is 2 (rule 1):
 *
 * ```
 * block {1, 3, 2, 4}
 *   slot[0] = (1, 2)  from offsets 0 and 2
 *   slot[1] = (3, 4)  from offsets 1 and 3
 * ```
 *
 * `x[1]` takes `{5, 7, 6, 8}` the same way. A name that runs out of values,
 * or has values left over, throws.
 */

namespace internal {

template <typename Scalar, typename Context>
inline auto get_values(const Context& context, const std::string& name) {
  if constexpr (std::is_same_v<Scalar, int>) {
    return context.vals_i(name);
  } else {
    return context.vals_r(name);
  }
}

template <typename T>
inline constexpr bool is_supported_scalar_v
    = std::is_arithmetic_v<T> || is_complex<T>::value;

template <typename T>
inline constexpr bool is_flat_vector_v
    = std::is_same_v<value_type_t<T>, scalar_type_t<T>>;

template <typename InVec>
using source_map_t = Eigen::Map<
    const Eigen::Matrix<value_type_t<InVec>, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::Unaligned, Eigen::InnerStride<Eigen::Dynamic>>;

template <typename T>
using flat_map_t
    = Eigen::Map<Eigen::Matrix<scalar_type_t<T>, Eigen::Dynamic, 1>>;

template <typename T, typename InVec>
inline void read_payload(T& x, const InVec& values, std::size_t offset,
                         std::size_t stride) {
  using map_t = source_map_t<InVec>;
  if constexpr (std::is_arithmetic_v<T>) {
    x = values[offset];
  } else if constexpr (is_eigen_v<T>) {
    x = map_t(values.data() + offset, x.rows(), x.cols(), stride);
  } else if constexpr (is_flat_vector_v<T>) {
    const std::size_t size = x.size();
    flat_map_t<T> dst(x.data(), size);
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
  } else if constexpr (is_flat_vector_v<T>) {
    const std::size_t size = x.size();
    flat_map_t<T> dst(x.data(), size);
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
  } else if constexpr (is_flat_vector_v<T>) {
    const std::size_t size = x.size();
    flat_map_t<T> dst(x.data(), size);
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
  } else if constexpr (is_flat_vector_v<T>) {
    const std::size_t size = x.size();
    flat_map_t<T> dst(x.data(), size);
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
  } else if constexpr (is_flat_vector_v<T>) {
    const std::size_t size = x.size();
    flat_map_t<T> dst(x.data(), size);
    dst.real() = map_t(values.data(), size, 1, 1);
    dst.imag() = map_t(values.data() + imaginary_offset, size, 1, 1);
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
          "read_from_context: ran out of values filling slot "
          + std::to_string(Slot + 1));
    }
    if (size > 0) {
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
 * Read a tuple-free variable out of a `var_context` into an already-sized
 * destination.
 *
 * The whole destination is one name and one whole buffer, so the buffer is
 * fetched once and must supply exactly the number of values the destination
 * holds. A complex destination takes two values per coefficient, every real
 * component before every imaginary one.
 *
 * @tparam T Destination type with no `std::tuple` anywhere inside: `int`,
 * `double`, `std::complex<double>`, an Eigen vector, row vector or matrix, or
 * a rectangular `std::vector` nesting of these. Its scalar type must be
 * `int`, `double` or `std::complex<double>`.
 * @tparam Context Source providing `vals_i(name)` and `vals_r(name)` as const
 * members returning `std::vector<int>` and `std::vector<double>` by value.
 * Integer destinations read `vals_i`, real and complex ones read `vals_r`.
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
          require_not_t<contains_tuple<T>>* = nullptr>
inline void read_from_context(T& x, const Context& context,
                              const std::string& name) {
  using scalar_t = scalar_type_t<T>;
  static_assert(
      internal::is_supported_scalar_v<scalar_t>,
      "read_from_context requires int, double or complex<double> scalars");
  const auto values = internal::get_values<scalar_t>(context, name);
  const std::size_t size = math::num_elements(x);
  const std::size_t expected = is_complex<scalar_t>::value ? 2 * size : size;
  if (values.size() != expected) {
    throw std::runtime_error("read_from_context: " + name + " expected "
                             + std::to_string(expected) + " values, got "
                             + std::to_string(values.size()));
  }
  if (size == 0) {
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
  } else if constexpr (internal::is_flat_vector_v<T>) {
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
 * Read a variable containing `std::tuple` out of a `var_context` into an
 * already-sized destination.
 *
 * Every tuple slot is a separate name, so this reads one name per leaf of the
 * tuple skeleton, fetching and releasing each buffer before the next. If a
 * later name fails, the names already read remain written. See the file
 * comment above for a worked example of the two traversals.
 *
 * Call it with three arguments; `SlotT` and `path` carry the recursion and
 * default to the root of the tuple skeleton and the empty path.
 *
 * @tparam T Destination type with a `std::tuple` somewhere inside: a
 * `std::tuple`, or a rectangular `std::vector` nesting around one. Every leaf
 * scalar must be `int`, `double` or `std::complex<double>`.
 * @tparam SlotT Position in the tuple skeleton, `scalar_type_t<T>` at the
 * root. A `std::tuple` here means the name has more slots below it; anything
 * else means the name is a leaf and is read. Never a type of `x` itself,
 * since `scalar_type_t` has erased the `std::vector` and Eigen layers.
 * @tparam Context Source providing `vals_i(name)` and `vals_r(name)` as const
 * members returning `std::vector<int>` and `std::vector<double>` by value.
 * @tparam Slots Tuple slot indices from the root to `SlotT`, excluding array
 * indices. The path that `fill_slots` follows through the destination.
 * @param[in,out] x Destination with every dimension already allocated. It is
 * always the root: the recursion descends `SlotT` and `path`, never `x`.
 * @param[in] context Source of the named integer and real buffers.
 * @param[in] name Variable name, with the dotted one-based suffix for `Slots`
 * already appended.
 * @param[in] path `Slots` as a value, so the pack is deduced rather than
 * given.
 * @throw std::runtime_error if a name supplies a number of values other than
 * the number its destinations hold. Exceptions from the context propagate
 * unchanged.
 */
template <typename T, typename SlotT = scalar_type_t<T>, typename Context,
          std::size_t... Slots, require_t<contains_tuple<T>>* = nullptr>
inline void read_from_context(T& x, const Context& context,
                              const std::string& name,
                              std::index_sequence<Slots...> path = {}) {
  if constexpr (is_tuple_v<SlotT>) {
    math::index_apply<std::tuple_size_v<SlotT>>(
        [&x, &context, &name](auto... Slot) {
          (read_from_context<T, std::tuple_element_t<Slot.value, SlotT>>(
               x, context, name + "." + std::to_string(Slot.value + 1),
               std::index_sequence<Slots..., Slot.value>{}),
           ...);
        });
  } else {
    static_assert(
        internal::is_supported_scalar_v<SlotT>,
        "read_from_context requires int, double or complex<double> scalars");
    const auto values = internal::get_values<SlotT>(context, name);
    std::size_t cursor = 0;
    internal::fill_slots(x, values, cursor, path);
    if (cursor != values.size()) {
      throw std::runtime_error("read_from_context: " + name + " supplied "
                               + std::to_string(values.size())
                               + " values but the destination holds "
                               + std::to_string(cursor));
    }
  }
}

}  // namespace io

}  // namespace stan

#endif
