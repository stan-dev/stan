#ifndef STAN__INTERFACE_CALLBACKS__WRITER__NOOP_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__NOOP_HPP

#include <stan/interface_callbacks/writer/writer.hpp>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      class noop: public writer {
      public:
        void write_key_value(const std::string& key, double value) {}
        void write_key_value(const std::string& key, const std::string& value) {}
        void write_state_names(std::vector<std::string>& names) {}
        void write_state(std::vector<double>& state) {}
        void write_message(const std::string& message) {}
      };

    }
  }
}

#endif
