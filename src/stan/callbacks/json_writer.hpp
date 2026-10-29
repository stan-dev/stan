#ifndef STAN_CALLBACKS_JSON_WRITER_HPP
#define STAN_CALLBACKS_JSON_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/io/json/write_json_value.hpp>
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
 * @tparam Stream A type with a valid `operator<<(std::string)` and `flush()`
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
   * Writes key plus colon for key-value pair.
   *
   * @param[in] key member name.
   */
  void write_key(const std::string& key) {
    stan::json::write_key(output_, key, record_depth_);
  }

  template <typename T>
  void write_value(const std::string& key, const T& value) {
    if (output_ == nullptr) {
      return;
    }
    write_sep();
    write_key(key);
    stan::json::write_json_value(output_, value, record_depth_);
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
      output_->flush();
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
    write_value(key, value);
  }

  /**
   * Write a key-value pair where the value is a const char*.
   * @param key Name of the value pair
   * @param value pointer to chars to write.
   */
  void write(const std::string& key, const char* value) {
    write_value(key, value);
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
  void write(const std::string& key, int value) { write_value(key, value); }

  /// Write a key-value pair with an unsigned long value.
  void write(const std::string& key,
             unsigned long value) {  // NOLINT(runtime/int)
    write_value(key, value);
  }

  /// Write a key-value pair with an unsigned long long value.
  void write(const std::string& key,
             unsigned long long value) {  // NOLINT(runtime/int)
    write_value(key, value);
  }

  /**
   * Write a key-value pair where the value is an `long long int`.
   * @param key Name of the value pair
   * @param value `long long int` to write.
   */
  void write(const std::string& key,
             long long int value  // NOLINT(runtime/int)
  ) {
    write_value(key, value);
  }

  /**
   * Write a key-value pair where the value is an `unsigned int`.
   * @param key Name of the value pair
   * @param value `unsigned int` to write.
   */
  void write(const std::string& key, unsigned int value) {
    write_value(key, value);
  }

  /**
   * Write a key-value pair where the value is a double.
   * @param key Name of the value pair
   * @param value double to write.
   */
  void write(const std::string& key, double value) { write_value(key, value); }

  /**
   * Write a key-value pair where the value is a complex value.
   * @param key Name of the value pair
   * @param value complex value to write.
   */
  void write(const std::string& key, const std::complex<double>& value) {
    write_value(key, value);
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<std::string>& values) {
    write_value(key, values);
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<double>& values) {
    write_value(key, values);
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<int>& values) {
    write_value(key, values);
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key,
             const std::vector<std::complex<double>>& values) {
    write_value(key, values);
  }

  /**
   * Write a key-value pair where the value is an Eigen Vector.
   * @param key Name of the value pair
   * @param vec Eigen Vector to write.
   */
  void write(const std::string& key, const Eigen::VectorXd& vec) {
    write_value(key, vec);
  }

  /**
   * Write a key-value pair where the value is an Eigen Vector.
   * @param key Name of the value pair
   * @param vec Eigen Vector to write.
   */
  void write(const std::string& key, const Eigen::RowVectorXd& vec) {
    write_value(key, vec);
  }

  /**
   * Write a key-value pair where the value is an Eigen Matrix.
   * @param key Name of the value pair
   * @param mat Eigen Matrix to write.
   */
  void write(const std::string& key, const Eigen::MatrixXd& mat) {
    write_value(key, mat);
  }
};

}  // namespace callbacks
}  // namespace stan
#endif
