#ifndef STAN_INTERFACE_CALLBACKS_WRITER_NOOP_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_NOOP_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      class noop: public base_writer {
      public:
        template <class T>
        void operator()(const std::vector<T>& x) {}
        void operator()(const std::string x) {}
        void operator()() {}
        bool is_writing() const { return false; }
      };

    }
  }
}

#endif
