#ifndef STAN__INTERFACE_CALLBACKS__WRITER__NOOP_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__NOOP_HPP

#include <stan/interface_callbacks/writer/writer.hpp>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      class noop: public writer {
      public:
        void writer_key_value(const std::string& key, double value) {}
        void writer_key_value(const std::string& key, const std::string& value) {}
        void write_state_names(const std::vector<std::string>& names) {}
        void write_state(const std::vector<double>& state) {}
        void write_message(const std::string& message) {}
      };

    }
  }
}

#endif
