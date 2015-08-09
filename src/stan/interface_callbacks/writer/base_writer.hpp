#ifndef STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP

#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      class base_writer {
      public:
        virtual ~base_writer() {}
        // Can't enforce this method with a pure virtual function
        // template <class T>
        // virtual void operator()(const std::vector<T>& x) = 0;
        virtual void operator()(const std::string x) = 0;
        virtual void operator()() = 0;
        virtual bool is_writing() const = 0;
      };

    }
  }
}

#endif
