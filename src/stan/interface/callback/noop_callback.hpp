#ifndef STAN_INTERFACE_CALLBACK_NOOP_CALLBACK_HPP
#define STAN_INTERFACE_CALLBACK_NOOP_CALLBACK_HPP

#include <stan/interface/callback/callback.hpp>

namespace stan {
  namespace interface {
    namespace callback {

      class noop_callback: public callback {
      public:
        noop_callback() {}
        void operator()() { };
      };
    
    }
  }
}

#endif
