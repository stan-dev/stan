#ifndef STAN__INTERFACE_CALLBACKS__INTERRUPT__NOOP_HPP
#define STAN__INTERFACE_CALLBACKS__INTERRUPT__NOOP_HPP

#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>

namespace stan {
  namespace interface_callbacks {
    namespace interrupt {

      class noop: public base_interrupt {
      public:
        noop() {}
        void operator()() { };
      };
    
    }
  }
}

#endif
