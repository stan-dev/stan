#ifndef STAN_CALLBACKS_INTERRUPT_NOOP_HPP
#define STAN_CALLBACKS_INTERRUPT_NOOP_HPP

#include <stan/callbacks/interrupt/base_interrupt.hpp>

namespace stan {
  namespace callbacks {
    namespace interrupt {

      class noop: public base_interrupt {
      public:
        noop() {}
        void operator()() { }
      };

    }
  }
}

#endif
