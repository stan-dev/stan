#ifndef STAN_CALLBACKS_INTERRUPT_NOOP_INTERRUPT_HPP
#define STAN_CALLBACKS_INTERRUPT_NOOP_INTERRUPT_HPP

#include <stan/callbacks/interrupt.hpp>

namespace stan {
  namespace callbacks {

    /**
     * No op interrupt.
     *
     * This is a trivial implementation of the interrupt that
     * does nothing.
     */
    class noop_interrupt: public interrupt {
    public:
      noop_interrupt() {}
      void operator()() { }
    };

  }
}
#endif
