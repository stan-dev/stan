#ifndef STAN_CALLBACKS_JSON_WRITER_HPP
#define STAN_CALLBACKS_JSON_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <ostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

namespace stan {
namespace callbacks {

/**
 * The `json_writer` callback is used output a single JSON object.
 * A JSON object is a mapping from element names to values which can be
 * either a scalar or array element, or a nested JSON object.
 * Objects are output elementwise via `write` callbacks which send
 * key, value pairs to the output stream.  Because JSON format
 * requires a comma between elements, the writer maintains
 * internal state to determine whether or not to output the comma separator.
 * The writer doesn't try to validate the object's internal structure
 * or object completeness, only syntactic correctness.
 *
 * @tparam Stream A type with with a valid `operator<<(std::string)`
 * @tparam Deleter A class with a valid `operator()` method for deleting the
 * output stream
 */
template <typename Stream, typename Deleter = std::default_delete<Stream>>
class json_writer final : public structured_writer {
 private:
  // Output stream
  std::unique_ptr<Stream, Deleter> output_{nullptr};
  // Whether or not the record's current object needs a comma separator
  bool record_element_needs_comma_ = false;
  // Depth of records (used to determine whether or not to print comma
  // separator)
  int record_depth_ = 0;

  /**
   * Determines whether a record's internal object requires a comma separator
   */
  void write_sep() {
    if (record_element_needs_comma_) {
      *output_ << ",";
    } else {
      record_element_needs_comma_ = true;
    }
    *output_ << "\n";
  }

  /**
   * Process a string to escape special characters.
   * Valid json strings cannot contain any of the special characters
   * `'\\', '"', '/', '\b', '\f', '\n', '\r', '\t', '\v', '\a', '\0'`.
   * In order to print these characters, they must be escaped.
   * @param value The string to process.
   * @return The processed string.
   */
  std::string process_string(const std::string& value) {
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

  /**
   * Writes key plus colon for key-value pair.
   *
   * @param[in] key member name.
   */
  void write_key(const std::string& key) {
    *output_ << std::string(record_depth_ * 2, ' ') << "\""
             << process_string(key) << "\" : ";
  }

  template <typename T>
  void write_int_like(const std::string& key, T value) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    *output_ << value;
  }

  /**
   * Writes a single value.  Corrects capitalization for inf and nans.
   *
   * @param[in] v value
   */
  void write_value(double v) {
    if (unlikely(std::isinf(v))) {
      if (v > 0) {
        *output_ << "Inf";
      } else {
        *output_ << "-Inf";
      }
    } else if (unlikely(std::isnan(v))) {
      *output_ << "NaN";
    } else {
      *output_ << v;
    }
  }

  /**
   * Writes a single complex value.
   *
   * @param[in] v value
   */
  void write_complex_value(std::complex<double> v) {
    *output_ << "[";
    write_value(v.real());
    *output_ << ", ";
    write_value(v.imag());
    *output_ << "]";
  }

  /**
   * Writes the set of comma separated values in an Eigen (row) vector.
   *
   * @param[in] v Values in an `Eigen::Vector`
   */
  template <typename Derived>
  void write_eigen_vector(const Eigen::DenseBase<Derived>& v) {
    *output_ << "[ ";
    if (v.size() > 0) {
      size_t last = v.size() - 1;
      for (Eigen::Index i = 0; i < last; ++i) {
        write_value(v[i]);
        *output_ << ", ";
      }
      write_value(v[last]);
    }
    *output_ << " ]";
  }

 public:
  /**
   * Constructs a no-op json writer.
   *
   */
  json_writer() : output_(nullptr) {}

  /**
   * Constructs a json writer with an output stream.
   *
   * @param[in, out] output unique pointer to a type inheriting from
   * `std::ostream`
   */
  explicit json_writer(std::unique_ptr<Stream, Deleter>&& output)
      : output_(std::move(output)) {}

  /** copy constructor */
  json_writer(json_writer& other) = delete;

  /** move constructor */
  json_writer(json_writer&& other) noexcept
      : output_(std::move(other.output_)) {}

  virtual ~json_writer() {}

  /**
   * Writes "{", initial token of a JSON record.
   */
  void begin_record() {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    *output_ << "{";
    record_depth_++;
    record_element_needs_comma_ = false;
  }

  /**
   * Writes "\"key\" : {", initial token of a named JSON record.
   * @param[in] key The name of the record.
   */
  void begin_record(const std::string& key) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    *output_ << "{";
    record_depth_++;
    record_element_needs_comma_ = false;
  }
  /**
   * Writes "}", final token of a JSON record.
   */
  void end_record() {
    if (output_ == nullptr) {
      return;
    }
    record_depth_--;
    *output_ << "\n" << std::string(record_depth_ * 2, ' ') << "}";
    if (record_depth_ > 0) {
      record_element_needs_comma_ = true;
    } else {
      *output_ << "\n";
    }
  }

  /**
   * Write a key-value pair to the output stream with a value of null as the
   * value.
   * @param key Name of the value pair
   */
  void write(const std::string& key) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    *output_ << "null";
  }

  /**
   * Write a key-value pair where the value is a string.
   * @param key Name of the value pair
   * @param value string to write.
   */
  void write(const std::string& key, const std::string& value) {
    if (output_ == nullptr) {
      return;
    }
    std::string processsed_string = process_string(value);
    write_sep();
    write_key(key);
    *output_ << "\"" << processsed_string << "\"";
  }

  /**
   * Write a key-value pair where the value is a const char*.
   * @param key Name of the value pair
   * @param value pointer to chars to write.
   */
  void write(const std::string& key, const char* value) {
    if (output_ == nullptr) {
      return;
    }
    std::string processsed_string = process_string(value);
    write_sep();
    write_key(key);
    *output_ << "\"" << processsed_string << "\"";
  }

  /**
   * Write a key-value pair where the value is a bool.
   * @param key Name of the value pair
   * @param value bool to write.
   */
  void write(const std::string& key, bool value) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    *output_ << (value ? "true" : "false");
  }

  /**
   * Write a key-value pair where the value is an int.
   * @param key Name of the value pair
   * @param value int to write.
   */
  void write(const std::string& key, int value) { write_int_like(key, value); }

  /**
   * Write a key-value pair where the value is an `std::size_t`.
   * @param key Name of the value pair
   * @param value `std::size_t` to write.
   */
  void write(const std::string& key, std::size_t value) {
    write_int_like(key, value);
  }

  /**
   * Write a key-value pair where the value is an `long long int`.
   * @param key Name of the value pair
   * @param value `long long int` to write.
   */
  void write(const std::string& key,
             long long int value  // NOLINT(runtime/int)
  ) {
    write_int_like(key, value);
  }

  /**
   * Write a key-value pair where the value is an `unsigned int`.
   * @param key Name of the value pair
   * @param value `unsigned int` to write.
   */
  void write(const std::string& key, unsigned int value) {
    write_int_like(key, value);
  }

  /**
   * Write a key-value pair where the value is a double.
   * @param key Name of the value pair
   * @param value double to write.
   */
  void write(const std::string& key, double value) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    write_value(value);
  }

  /**
   * Write a key-value pair where the value is a complex value.
   * @param key Name of the value pair
   * @param value complex value to write.
   */
  void write(const std::string& key, const std::complex<double>& value) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    write_complex_value(value);
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<std::string>& values) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);

    *output_ << "[ ";
    if (values.size() > 0) {
      auto last = values.end();
      --last;
      for (auto it = values.begin(); it != last; ++it) {
        *output_ << process_string(*it) << ", ";
      }
    }
    *output_ << values.back() << " ]";
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<double>& values) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);

    *output_ << "[ ";
    if (values.size() > 0) {
      auto last = values.end();
      --last;
      for (auto it = values.begin(); it != last; ++it) {
        write_value(*it);
        *output_ << ", ";
      }
      write_value(values.back());
    }
    *output_ << " ]";
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<int>& values) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);

    *output_ << "[ ";
    if (values.size() > 0) {
      auto last = values.end();
      --last;
      for (auto it = values.begin(); it != last; ++it) {
        *output_ << *it << ", ";
      }
    }
    *output_ << values.back() << " ]";
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key,
             const std::vector<std::complex<double>>& values) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);

    *output_ << "[ ";
    if (values.size() > 0) {
      size_t last = values.size() - 1;
      for (size_t i = 0; i < last; ++i) {
        write_complex_value(values[i]);
        *output_ << ", ";
      }
      write_complex_value(values[last]);
    }
    *output_ << " ]";
  }

  /**
   * Write a key-value pair where the value is an Eigen Vector.
   * @param key Name of the value pair
   * @param vec Eigen Vector to write.
   */
  void write(const std::string& key, const Eigen::VectorXd& vec) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    write_eigen_vector(vec);
  }

  /**
   * Write a key-value pair where the value is an Eigen Vector.
   * @param key Name of the value pair
   * @param vec Eigen Vector to write.
   */
  void write(const std::string& key, const Eigen::RowVectorXd& vec) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    write_eigen_vector(vec);
  }

  /**
   * Write a key-value pair where the value is an Eigen Matrix.
   * @param key Name of the value pair
   * @param mat Eigen Matrix to write.
   */
  void write(const std::string& key, const Eigen::MatrixXd& mat) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    *output_ << "[ ";
    if (mat.rows() > 0) {
      Eigen::Index last = mat.rows() - 1;
      for (Eigen::Index i = 0; i < last; ++i) {
        write_eigen_vector(mat.row(i));
        *output_ << ", ";
      }
      write_eigen_vector(mat.row(last));
    }
    *output_ << " ]";
  }
};

}  // namespace callbacks
}  // namespace stan
#endif
