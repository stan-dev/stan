#ifndef STAN_CALLBACKS_TABLE_WRITER_HPP
#define STAN_CALLBACKS_TABLE_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <vector>
#include <string>

namespace stan {
namespace callbacks {

/**
 * <code>table_writer</code> is a base class defining the interface
 * for tabular data callbacks. It can be used as a no-op implementation.
 */
class table_writer {
 public:
  /**
   * Virtual destructor.
   */
  virtual ~table_writer() {}

  /**
   * Process table header
   */
  virtual void begin_header() {}

  /**
   * Writes a vector of column names for a data table.
   * @param values vector of strings to write.
   */
  virtual void write_header(const std::vector<std::string>& values) {}

  /**
   * Process table header
   */
  virtual void end_header() {}

  /**
   * Process table row
   */
  virtual void begin_row() {}

  /**
   * Process table row
   */
  virtual void end_row() {}

  /**
   * Process table row
   */
  virtual void pad_row() {}

  /**
   * Write a series of real-valued data table values
   * @param values vector of values to write.
   */
  virtual void write_flat(const std::vector<double>& values) {}

  /**
   * Write a series of int-values data table values
   * @param values vector of values to write.
   */
  virtual void write_flat(const std::vector<int>& values) {}

};

}  // namespace callbacks
}  // namespace stan
#endif
