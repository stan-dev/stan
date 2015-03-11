#ifndef STAN__INTERFACE_CALLBACKS__INTERRUPT__INTERRUPT_HPP
#define STAN__INTERFACE_CALLBACKS__INTERRUPT__INTERRUPT_HPP

namespace stan {
  namespace interface_callbacks {
    namespace callbacks {

      // This callback allows the interfaces to interrupt
      // execution at various points in command.hpp
      class interrupt {
      public:
        interrupt() {}
        virtual void operator()() = 0;
      };
    
    }
  }
}

#endif
