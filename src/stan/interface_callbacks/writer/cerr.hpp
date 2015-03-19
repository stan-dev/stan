#ifndef STAN__INTERFACE_CALLBACKS__WRITER__CERR_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__CERR_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <iostream>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      // FIXME: Move to CmdStan
      class cerr: public base_writer {
      public:
        void write_key_value(const std::string& key, double value) {
          std::cerr << key << " = " << value << std::endl;
        }
        void write_key_value(const std::string& key, const std::string& value) {
          std::cerr << key << " = " << value << std::endl;
        }
        
        void write_state_names(std::vector<std::string>& names) {}
        void write_state(std::vector<double>& state) {}
        
        void write_message() {
          std::cerr << std::endl;
        }
        void write_message(const std::string& message) {
          std::cerr << message << std::endl;
        };
      };

    }
  }
}

#endif
