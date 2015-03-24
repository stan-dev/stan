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
        void operator()(const std::string& key, double value) {
          std::cerr << key << " = " << value << std::endl;
        }
        void operator()(const std::string& key, const std::string& value) {
          std::cerr << key << " = " << value << std::endl;
        }
        
        void operator()(std::vector<std::string>& names) {}
        void operator()(std::vector<double>& state) {}
        
        void operator()() {
          std::cerr << std::endl;
        }
        void operator()(const std::string& message) {
          std::cerr << message << std::endl;
        };
      };

    }
  }
}

#endif
