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
   * Write start symbol
   */
  virtual void start_token() {}

  /**
   * Write end symbol
   *
   */
  virtual void end_token() {}

  virtual void keyed_begin(const std::string& key) {}
  virtual void keyed_null() {}
  virtual void keyed_bool(const std::string& key, bool value) {}

  virtual void keyed_value(const std::string& key, int value) {}
  virtual void keyed_value(const std::string& key, double value) {}
  virtual void keyed_value(const std::string& key,
    const std::tuple<Eigen::VectorXd, Eigen::VectorXd>& value) {}

  virtual void keyed_values(const std::string& key,
                            const std::vector<double> value) {}
  virtual void keyed_values(const std::string& key,
    const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>& values) {}

  virtual void reset() {}

};

}  // namespace callbacks
}  // namespace stan
#endif
