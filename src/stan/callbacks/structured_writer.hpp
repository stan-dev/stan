#ifndef STAN_CALLBACKS_STRUCTURED_WRITER_HPP
#define STAN_CALLBACKS_STRUCTURED_WRITER_HPP

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
};

}  // namespace callbacks
}  // namespace stan
#endif
