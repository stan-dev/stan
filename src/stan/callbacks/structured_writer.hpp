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

  virtual void begin() {}
  virtual void end() {}
  virtual void begin_list() {}
  virtual void end_list() {}

  virtual void write_begin(const std::string& key) {}
  virtual void write(const std::string& key) {}
  virtual void write(const std::string& key, const std::string& value) {}
  virtual void write() {}
  virtual void write(const std::string& key, bool value) {}

  virtual void write(const std::string& key, int value) {}
  virtual void write(const std::string& key, std::size_t value) {}
  virtual void write(const std::string& key, double value) {}
  virtual void write(const std::string& key, const std::complex<double>& value) {}
  virtual void write(const std::string& key,
                            const std::vector<double> value) {}
  void write(const std::string& key, const std::vector<std::string>& values) {}
  void write(const std::string& key, const Eigen::MatrixXd& mat) {}
  void write(const std::string& key, const Eigen::VectorXd& vec) {}
  void write(const std::string& key, const Eigen::RowVectorXd& vec) {}

  virtual void reset() {}
};

}  // namespace callbacks
}  // namespace stan
#endif
