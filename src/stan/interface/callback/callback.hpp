#ifndef STAN_INTERFACE_CALLBACK_CALLBACK_HPP
#define STAN_INTERFACE_CALLBACK_CALLBACK_HPP

namespace stan {
  namespace interface {
    namespace callback {

      class callback {
      public:
        callback() {}
        virtual void operator()() = 0;
      };
    
    }
  }
}

#endif
