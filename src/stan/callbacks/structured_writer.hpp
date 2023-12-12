#ifndef STAN_CALLBACKS_STRUCTURED_WRITER_HPP
#define STAN_CALLBACKS_STRUCTURED_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <vector>
#include <string>

namespace stan {
namespace callbacks {

/**
 * <code>structured_writer</code> is a base class defining the interface
 * for Stan structured_writer callbacks. The base class can be used as a
 * no-op implementation.
 */
class structured_writer {
 public:
  /**
   * Virtual destructor.
   */
  virtual ~structured_writer() {}

  /**
   * Writes start token of a structured record.
   */
  virtual void begin_record() {}

  /**
   * Writes key followed by start token of a structured record.
   * @param[in] key The name of the record.
   */
  virtual void begin_record(const std::string& key) {}

  /**
   * Writes end token of a structured record.
   */
  virtual void end_record() {}

  /**
   * Write a key-value pair to the output stream with a value of null as the
   * value.
   * @param key Name of the value pair
   */
  virtual void write(const std::string& key) {}

  /**
   * Write a key-value pair where the value is a string.
   * @param key Name of the value pair
   * @param value string to write.
   */
  virtual void write(const std::string& key, const std::string& value) {}

  /**
   * Write a key-value pair where the value is a bool.
   * @param key Name of the value pair
   * @param value bool to write.
   */
  virtual void write(const std::string& key, bool value) {}

  /**
   * Write a key-value pair where the value is an int.
   * @param key Name of the value pair
   * @param value int to write.
   */
  virtual void write(const std::string& key, int value) {}

  /**
   * Write a key-value pair where the value is an `std::size_t`.
   * @param key Name of the value pair
   * @param value `std::size_t` to write.
   */
  virtual void write(const std::string& key, std::size_t value) {}

  /**
   * Write a key-value pair where the value is an `long long int`.
   * @param key Name of the value pair
   * @param value `long long int` to write.
   */
  virtual void write(const std::string& key,
                     long long int value  // NOLINT(runtime/int)
  ) {}

  /**
   * Write a key-value pair where the value is an `unsigned int`.
   * @param key Name of the value pair
   * @param value `unsigned int` to write.
   */
  virtual void write(const std::string& key, unsigned int value) {}

  /**
   * Write a key-value pair where the value is a double.
   * @param key Name of the value pair
   * @param value double to write.
   */
  virtual void write(const std::string& key, double value) {}

  /**
   * Write a key-value pair where the value is a complex value.
   * @param key Name of the value pair
   * @param value complex value to write.
   */
  virtual void write(const std::string& key,
                     const std::complex<double>& value) {}
  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  virtual void write(const std::string& key,
                     const std::vector<double>& values) {}

  /**
   * Write a key-value pair where the value is a vector of strings to be made a
   * list.
   * @param key Name of the value pair
   * @param values vector of strings to write.
   */
  virtual void write(const std::string& key,
                     const std::vector<std::string>& values) {}

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  virtual void write(const std::string& key,
                     const std::vector<std::complex<double>>& values) {}

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  virtual void write(const std::string& key, const std::vector<int>& values) {}

  /**
   * Write a key-value pair where the value is an Eigen Matrix.
   * @param key Name of the value pair
   * @param mat Eigen Matrix to write.
   */
  virtual void write(const std::string& key, const Eigen::MatrixXd& mat) {}

  /**
   * Write a key-value pair where the value is an Eigen Vector.
   * @param key Name of the value pair
   * @param vec Eigen Vector to write.
   */
  virtual void write(const std::string& key, const Eigen::VectorXd& vec) {}

  /**
   * Write a key-value pair where the value is a Eigen RowVector.
   * @param key Name of the value pair
   * @param vec Eigen RowVector to write.
   */
  virtual void write(const std::string& key, const Eigen::RowVectorXd& vec) {}

  /**
   * Write a key-value pair where the value is a const char*.
   * @param key Name of the value pair
   * @param value pointer to chars to write.
   */
  virtual void write(const std::string& key, const char* value) {}

  /**
   * Writes a vector of column names for a data table.
   * @param values vector of strings to write.
   */
  virtual void table_header(const std::vector<std::string>& values) {}

  /**
   * Writes a vector of values as a row of the data table.
   * @param values vector of strings to write.
   */
  virtual void table_row(const std::vector<double>& values) {}



 protected:
  /**
   * Process a string to escape special characters.
   * Valid csv strings cannot contain any of the special characters
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








};

}  // namespace callbacks
}  // namespace stan
#endif
