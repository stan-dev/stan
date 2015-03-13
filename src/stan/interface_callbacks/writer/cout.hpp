#ifndef STAN__INTERFACE_CALLBACKS__WRITER__COUT_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__COUT_HPP

#include <stan/interface_callbacks/writer/writer.hpp>
#include <iostream>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      // FIXME: Move to CmdStan
      class cout: public writer {
      public:
        void write_key_value(const std::string& key, double value) {}
        void write_key_value(const std::string& key, const std::string& value) {}
        void write_state_names(std::vector<std::string>& names) {}
        void write_state(std::vector<double>& state) {}
        void write_message(const std::string& message) {
          std::cout << message << std::endl;
        }
      };

    }
  }
}

#endif
