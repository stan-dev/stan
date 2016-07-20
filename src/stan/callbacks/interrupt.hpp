#ifndef STAN_CALLBACKS_INTERRUPT_HPP
#define STAN_CALLBACKS_INTERRUPT_HPP

namespace stan {
  namespace callbacks {

    class interrupt {
    public:
      interrupt() {}
      virtual void operator()() = 0;
    };

  }
}
#endif
