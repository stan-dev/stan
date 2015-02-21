#ifndef STAN__INTERFACE__CALLBACK__CALLBACK_HPP
#define STAN__INTERFACE__CALLBACK__CALLBACK_HPP

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
