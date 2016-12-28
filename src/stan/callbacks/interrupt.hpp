#ifndef STAN_CALLBACKS_INTERRUPT_HPP
#define STAN_CALLBACKS_INTERRUPT_HPP

namespace stan {
  namespace callbacks {

    /**
     * <code>interrupt</code> is an abstract base class defining the interface
     * for Stan interrupt callbacks.
     *
     * The interrupt is called from within Stan algorithms to allow
     * for the interfaces to handle interrupt signals (ctrl-c).
     */
    class interrupt {
    public:
      interrupt() {}
      virtual void operator()() {
      }

      virtual ~interrupt() {}
    };

  }
}
#endif
