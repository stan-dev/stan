#ifndef STAN__INTERFACE__RECORDER__NOOP_HPP
#define STAN__INTERFACE__RECORDER__NOOP_HPP

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
