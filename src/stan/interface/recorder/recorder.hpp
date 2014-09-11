#ifndef STAN__INTERFACE__RECORDER__RECORDER_HPP
#define STAN__INTERFACE__RECORDER__RECORDER_HPP

#include <string>
#include <vector>

namespace stan {
  namespace interface {
    namespace recorder {
      
      class recorder {
      public:
        // Can't enforce this method with a pure virtual function
        //template <class T>
        //virtual void operator()(const std::vector<T>& x) = 0;
        virtual void operator()(const std::string x) = 0;
        virtual void operator()() = 0;
        virtual bool is_recording() const = 0;
      };

    }
  }
}

#endif
