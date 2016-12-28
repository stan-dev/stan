#ifndef STAN_CALLBACKS_NOOP_WRITER_HPP
#define STAN_CALLBACKS_NOOP_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace callbacks {

    /**
     * No op writer.
     *
     * This is a trivial implementation of the writer that
     * accepts input and writes nowhere.
     */
    class noop_writer: public writer {
    public:
      void operator()(const std::vector<std::string>& names) {}
      void operator()(const std::vector<double>& state) {}
      void operator()() {}
      void operator()(const std::string& message) {}
    };

  }
}
#endif
