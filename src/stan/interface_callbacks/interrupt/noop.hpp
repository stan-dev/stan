#ifndef STAN_INTERFACE_CALLBACKS_INTERRUPT_NOOP_HPP
#define STAN_INTERFACE_CALLBACKS_INTERRUPT_NOOP_HPP

#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>

namespace stan {
  namespace interface_callbacks {
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
