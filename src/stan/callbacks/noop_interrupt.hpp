#ifndef STAN_CALLBACKS_INTERRUPT_NOOP_INTERRUPT_HPP
#define STAN_CALLBACKS_INTERRUPT_NOOP_INTERRUPT_HPP

#include <stan/callbacks/interrupt.hpp>

namespace stan {
  namespace callbacks {

      class noop_interrupt: public interrupt {
      public:
        noop_interrupt() {}
        void operator()() { }
      };

  }
}

#endif
