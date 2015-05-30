#ifndef STAN_INTERFACE_RECORDER_RECORDER_HPP
#define STAN_INTERFACE_RECORDER_RECORDER_HPP

#include <string>
#include <vector>

namespace stan {
  namespace interface {
    namespace recorder {

      class recorder {
      public:
        virtual ~recorder() {};
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
