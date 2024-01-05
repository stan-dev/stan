#ifndef STAN_CALLBACKS_WRITER_HPP
#define STAN_CALLBACKS_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <string>
#include <vector>

namespace stan {
namespace callbacks {

/**
 * <code>writer</code> is a base class defining the interface
 * for Stan writer callbacks. The base class can be used as a
 * no-op implementation.
 */
class writer {
 public:
  /**
   * Virtual destructor.
   */
  virtual ~writer() {}

  /**
   * Writes a set of names.
   *
   * @param[in] names Names in a std::vector
   */
  virtual void operator()(const std::vector<std::string>& names) {}

  /**
   * Writes a set of values.
   *
   * @param[in] state Values in a std::vector
   */
  virtual void operator()(const std::vector<double>& state) {}

  /**
   * Writes blank input.
   */
  virtual void operator()() {}

  /**
   * Writes a string.
   *
   * @param[in] message A string
   */
  virtual void operator()(const std::string& message) {}

  /**
   * Writes multiple rows and columns of values in csv format.
   *
   * Note: the precision of the output is determined by the settings
   *  of the stream on construction.
   *
   * @param[in] values A matrix of values. The input is expected to have
   * parameters in the rows and samples in the columns. The matrix is then
   * transposed for the output.
   */
  virtual void operator()(
      const Eigen::Ref<Eigen::Matrix<double, -1, -1>>& values) {}
};

}  // namespace callbacks
}  // namespace stan
#endif
