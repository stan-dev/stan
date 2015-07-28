#ifndef STAN_INTERFACE_CALLBACKS_INTERRUPT_BASE_INTERRUPT_HPP
#define STAN_INTERFACE_CALLBACKS_INTERRUPT_BASE_INTERRUPT_HPP

namespace stan {
  namespace interface_callbacks {
    namespace interrupt {

      class base_interrupt {
      public:
        base_interrupt() {}
        virtual void operator()() = 0;
      };

    }
  }
}

#endif
