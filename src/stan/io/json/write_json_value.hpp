#ifndef STAN_IO_JSON_WRITE_JSON_VALUE_HPP
#define STAN_IO_JSON_WRITE_JSON_VALUE_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/functor/for_each.hpp>

#include <string>
#include <vector>
#include <cstring>

namespace stan {
namespace json {

namespace internal {
/**
 * Process a string to escape special characters.
 * Valid json strings cannot contain any of the special characters
 * `'\\', '"', '/', '\b', '\f', '\n', '\r', '\t', '\v', '\a', '\0'`.
 * In order to print these characters, they must be escaped.
 * @param value The string to process.
 * @return The processed string.
 */
inline std::string process_string(const std::string& value) {
  static constexpr std::array<char, 11> chars_to_escape
      = {'\\', '"', '/', '\b', '\f', '\n', '\r', '\t', '\v', '\a', '\0'};
  static constexpr std::array<const char*, 11> chars_to_replace
      = {"\\\\", "\\\"", "\\/", "\\b", "\\f", "\\n",
         "\\r",  "\\t",  "\\v", "\\a", "\\0"};
  // Replacing every value leads to 2x the size
  std::string new_value(value.size() * 2, 'x');
  std::size_t pos = 0;
  std::size_t count = 0;
  std::size_t prev_pos = 0;
  while ((pos = value.find_first_of(chars_to_escape.data(), pos, 10))
         != std::string::npos) {
    for (int i = prev_pos; i < pos; ++i) {
      new_value[i + count] = value[i];
    }
    int idx
        = strchr(chars_to_escape.data(), value[pos]) - chars_to_escape.data();
    new_value[pos + count] = chars_to_replace[idx][0];
    new_value[pos + count + 1] = chars_to_replace[idx][1];
    pos += 1;
    count++;
    prev_pos = pos;
  }
  for (int i = prev_pos; i < value.size(); ++i) {
    new_value[i + count] = value[i];
  }
  // Shrink any unused space
  new_value.resize(value.size() + count);
  return new_value;
}

}  // namespace internal

/**
 * Writes key plus colon for key-value pair.
 */
template <typename Stream>
void write_key(Stream&& output, const std::string& key, int record_depth = 0) {
  *output << std::string(record_depth * 2, ' ') << "\""
          << internal::process_string(key) << "\" : ";
}

/**
 * Write a floating point value to JSON. NaN and infinite values are
 * rendered with capitalization.
 */
template <typename Stream, typename T,
          stan::require_floating_point_t<T>* = nullptr>
void write_json_value(Stream&& output, T value, int /*record_depth*/ = 0) {
  if (unlikely(std::isinf(value))) {
    if (value > 0) {
      *output << "Inf";
    } else {
      *output << "-Inf";
    }
  } else if (unlikely(std::isnan(value))) {
    *output << "NaN";
  } else {
    *output << value;
  }
}

/**
 * Write an integer to JSON.
 */
template <typename Stream, typename T, stan::require_integral_t<T>* = nullptr>
void write_json_value(Stream&& output, T value, int /*record_depth*/ = 0) {
  *output << value;
}

/**
 * Write a complex number to JSON. We render this as a 2-element list.
 */
template <typename Stream, typename T, stan::require_complex_t<T>* = nullptr>
void write_json_value(Stream&& output, T&& value, int /*record_depth*/ = 0) {
  *output << "[";
  write_json_value(output, value.real());
  *output << ", ";
  write_json_value(output, value.imag());
  *output << "]";
}

/**
 * Write an Eigen (Row)Vector to JSON.
 */
template <typename Stream, typename T,
          stan::require_eigen_vector_t<T>* = nullptr>
void write_json_value(Stream&& output, T&& value, int /*record_depth*/ = 0) {
  *output << "[ ";
  if (value.size() > 0) {
    size_t last = value.size() - 1;
    for (Eigen::Index i = 0; i < last; ++i) {
      write_json_value(output, value[i]);
      *output << ", ";
    }
    write_json_value(output, value[last]);
  }
  *output << " ]";
}

/**
 * Write an Eigen Matrix to JSON.
 */
template <typename Stream, typename T,
          stan::require_eigen_matrix_dynamic_t<T>* = nullptr>
void write_json_value(Stream&& output, T&& value, int /*record_depth*/ = 0) {
  *output << "[ ";
  if (value.rows() > 0) {
    Eigen::Index last = value.rows() - 1;
    for (Eigen::Index i = 0; i < last; ++i) {
      write_json_value(output, value.row(i));
      *output << ", ";
    }
    write_json_value(output, value.row(last));
  }
  *output << " ]";
}

/**
 * Write a string to JSON.
 */
template <typename Stream>
void write_json_value(Stream&& output, const std::string& value,
                      int /*record_depth*/ = 0) {
  *output << "\"" << internal::process_string(value) << "\"";
}

template <typename Stream, typename T,
          stan::math::require_tuple_t<T>* = nullptr>
void write_json_value(Stream&& output, const T& value, int record_depth = 0);

/**
 * Write a vector of values to JSON.
 */
template <typename Stream, typename T>
void write_json_value(Stream&& output, const std::vector<T>& values,
                      int record_depth = 0) {
  *output << "[ ";
  if (values.size() > 0) {
    auto last = values.end();
    --last;
    for (auto it = values.begin(); it != last; ++it) {
      write_json_value(output, *it, record_depth);
      *output << ", ";
    }
    write_json_value(output, values.back(), record_depth);
  }
  *output << " ]";
}

/**
 * Write a tuple of values to JSON. Each element of the tuple gets an integer
 * key, starting at "1".
 */
template <typename Stream, typename T, stan::math::require_tuple_t<T>*>
void write_json_value(Stream&& output, const T& value, int record_depth) {
  *output << '{';
  size_t i = 0;
  stan::math::for_each(
      [&i, record_depth, output](auto&& elt) {
        if (i > 0) {
          *output << ", ";
        }
        i++;
        write_key(output, std::to_string(i), record_depth);
        write_json_value(output, elt, record_depth + 1);
      },
      value);
  *output << '}';
}

}  // namespace json
}  // namespace stan
#endif
