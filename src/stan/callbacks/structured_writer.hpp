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
};

}  // namespace callbacks
}  // namespace stan
#endif
