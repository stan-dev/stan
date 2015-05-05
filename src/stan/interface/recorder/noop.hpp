#ifndef STAN_INTERFACE_RECORDER_NOOP_HPP
#define STAN_INTERFACE_RECORDER_NOOP_HPP

#include <stan/interface/recorder/recorder.hpp>

namespace stan {
  namespace interface {
    namespace recorder {
      
      class noop: public recorder {
      public:

        template <class T>
        void operator()(const std::vector<T>& x) {};
        void operator()(const std::string x) {};
        void operator()() {};
        bool is_recording() const { return false; }
      };

    }
  }
}

#endif
