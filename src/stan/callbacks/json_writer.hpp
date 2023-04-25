#ifndef STAN_CALLBACKS_JSON_WRITER_HPP
#define STAN_CALLBACKS_JSON_WRITER_HPP

#include <stan/callbacks/structured_writer.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
namespace callbacks {

/**
 * <code>json_writer</code> is an implementation of
 * <code>structured_writer</code> that writes JSON format data to a stream.
 * @tparam Stream A type with with a valid `operator<<(std::string)`
 * @tparam Deleter A class with a valid `operator()` method for deleting the output stream
 */
template <typename Stream, typename Deleter = std::default_delete<Stream>>
class json_writer final : public structured_writer {
   private:
  // Output stream
  std::unique_ptr<Stream, Deleter> output_{nullptr};
  // Whether or not the record's current object needs a comma separator
  bool record_internal_needs_comma_ = false;
  // Depth of records (used to determine whether or not to print comma separator)
  int record_depth_ = 0;
  // Whether or not the record's parent object needs a comma separator
  bool record_needs_comma_;

  /**
   * Writes a comma separator for the record's parent object if needed.
   */
  void write_record_comma_if_needed() {
    if (record_depth_ > 0 && record_needs_comma_) {
      *output_ << ",";
    }
  }

  /**
   * Determines whether or a record's internal object requires a comma separator
   */
  void write_sep() {
    if (record_internal_needs_comma_) {
      *output_ << ", ";
    } else {
      record_internal_needs_comma_ = true;
    }
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
    std::string new_value = value;
    static constexpr std::array<char, 11> chars_to_escape = {'\\', '"', '/', '\b', '\f', '\n', '\r', '\t', '\v', '\a', '\0'};
    static constexpr std::array<const char*, 11> chars_to_replace = {"\\\\", "\\\"", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t", "\\v", "\\a", "\\0"};
    for (int i = 0; i < 11; ++i) {
      std::size_t pos = 0;
      while ((pos = new_value.find(chars_to_escape[i], pos)) != std::string::npos) {
        new_value.replace(pos, 1, chars_to_replace[i]);
        pos += 2;
      }
    }
    return new_value;
  }

  /**
   * Writes key plus colon for key-value pair.
   *
   * @param[in] key member name.
   */
  void write_key(const std::string& key) { *output_ << "\"" << key << "\" : "; }
  /**
   * Writes a set of comma separated values.
   *
   * @param[in] v Values in a std::vector
   */
  void write_vector(const std::vector<std::string>& v) {
    if (v.empty()) {
      return;
    }
    *output_ << "[ ";
    auto last = v.end();
    --last;
    for (auto it = v.begin(); it != last; ++it) {
      *output_ << process_string(*it) << ", ";
    }
    *output_ << v.back() << " ]";
  }

  /**
   * Writes a set of comma separated values.
   *
   * @param[in] v Values in a std::vector
   */
  template <class T>
  void write_vector(const std::vector<T>& v) {
    if (v.empty()) {
      return;
    }
    *output_ << "[ ";
    auto last = v.end();
    --last;
    for (auto it = v.begin(); it != last; ++it) {
      *output_ << *it << ", ";
    }
    *output_ << v.back() << " ]";
  }
  
 public:
 json_writer() : output_(nullptr) {}
  /**
   * Constructs a json writer with an output stream.
   *
   * @param[in, out] A unique pointer to a type inheriting from `std::ostream`
   */
  explicit json_writer(std::unique_ptr<Stream, Deleter>&& output)
      : output_(std::move(output)) {
    if (output_ == nullptr)
      throw std::invalid_argument("writer cannot be null");
  }

  json_writer(json_writer& other) = delete;
  json_writer(json_writer&& other) : output_(std::move(other.output_)) {}

  virtual ~json_writer() {}

  /**
   * Writes "{", initial token of a JSON record.
   */
    void begin_record() {
        write_record_comma_if_needed();
        *output_ << "{";
        record_needs_comma_ = false;
        record_depth_++;
    }
    void begin_record(const std::string& key) {
        write_record_comma_if_needed();
        *output_ << "\"" << key << "\": {";
        record_needs_comma_ = false;
        record_depth_++;
    }
  /**
   * Writes "}", final token of a JSON record.
   */
  void end_record() {
      if (record_depth_ > 0) {
          *output_ << "}";
          record_depth_--;
          if (record_depth_ > 0) {
              record_needs_comma_ = true;
          }
      record_internal_needs_comma_ = false;
      } else {
          throw std::runtime_error("Attempted to end record, but there is no open record.");
      }
  }

  /**
   * Writes "[", initial token of a JSON list.
   */
  void begin_list() {
    *output_ << "[";
    record_internal_needs_comma_ = false;
  }
  /**
   * Writes "]", final token of a JSON list.
   */
  void end_list() {
    *output_ << "]";
    write_sep();
    *output_ << "\n";
  }

  /**
   * Write a key-value pair to the output stream with a value of null as the value. 
   * @param key The key to write
   */
  void write(const std::string& key) {
    write_sep();
    write_key(key);
    *output_ << "\"null\" ";
  }

  /**
   * Write a key-value pair where the value is a string.
   * @param value string to write.  
   */
  void write(const std::string& key, const std::string& value) {
    std::string processsed_string = process_string(value);
    write_sep();
    write_key(key);
    *output_ << "\"" << processsed_string << "\"";
  }

  /**
   * Write a key-value pair where the value is a string.
   * @param value string to write.  
   */
  void write(const std::string& key, bool value) {
    write_sep();
    write_key(key);
    *output_ << (value ? "true" : "false");
  }

  /**
   * Write a key-value pair where the value is a string.
   * @param value string to write.  
   */
  void write(const std::string& key, int value) {
    write_sep();
    write_key(key);
    *output_ << value;
  }

  /**
   * Write a key-value pair where the value is a string.
   * @param value string to write.  
   */
  void write(const std::string& key, std::size_t value) {
    write_sep();
    write_key(key);
    *output_ << value;
  }

  /**
   * Write a key-value pair where the value is a string.
   * @param value string to write.  
   */
  void write(const std::string& key, double value) {
    write_sep();
    write_key(key);
    *output_ << value;
  }

  
  /**
   * Write a key-value pair where the value is a string.
   * @param value string to write.  
   */
  void write(const std::string& key, const std::complex<double>& value) {
    write_sep();
    write_key(key);
      *output_ << "\"" << key << "\" : [" << value.real() << ", " << value.imag() << "]";
    }

  /**
   * Write a key-value pair where the value is a string.
   * @param value string to write.  
   */
  void write(const std::string& key, const std::vector<double>& values) {
    write_sep();
    write_key(key);
    write_vector(values);
  }

  /**
   * Write a key-value pair where the value is a vector of strings.
   * @param value vector of strings to write.  
   */
  void write(const std::string& key, const std::vector<std::string>& values) {
    write_sep();
    write_key(key);
    write_vector(values);
  }

  /**
   * Write a key-value pair where the value is a Eigen Matrix.
   * @param value Eigen Matrix to write.  
   */
  void write(const std::string& key, const Eigen::MatrixXd& mat) {
    write_sep();
    write_key(key);
    Eigen::IOFormat json_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "[", "]", "[", "]");
    *output_ << mat.format(json_format);
  }

  /**
   * Write a key-value pair where the value is an Eigen Vector.
   * @param value Eigen Vector to write.  
   */
  void write(const std::string& key, const Eigen::VectorXd& vec) {
    write_sep();
    write_key(key);
    Eigen::IOFormat json_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "", "", "", "[", "]");
    *output_ << vec.transpose().format(json_format);
  }

  /**
   * Write a key-value pair where the value is a Eigen RowVector.
   * @param value Eigen RowVector to write.  
   */
  void write(const std::string& key, const Eigen::RowVectorXd& vec) {
    write_sep();
    write_key(key);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "", "", "", "[", "]");
    *output_ << vec.format(CommaInitFmt);
  }

  /**
   * Reset state
   */
  void reset() { 
    record_internal_needs_comma_ = false; 
    record_needs_comma_ = false; 
    record_depth_ = 0;
  }

};

}  // namespace callbacks
}  // namespace stan
#endif
