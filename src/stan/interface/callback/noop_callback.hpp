#ifndef STAN__INTERFACE__CALLBACK__NOOP_CALLBACK_HPP
#define STAN__INTERFACE__CALLBACK__NOOP_CALLBACK_HPP

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
