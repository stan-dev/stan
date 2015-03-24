#ifndef STAN__INTERFACE_CALLBACKS__WRITER__COUT_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__COUT_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <iostream>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      // FIXME: Move to CmdStan
      class cout: public base_writer {
      public:
        void operator()(const std::string& key, double value) {
          std::cout << key << " = " << value << std::endl;
        }
        void operator()(const std::string& key, const std::string& value) {
          std::cout << key << " = " << value << std::endl;
        }
        
        void operator()(std::vector<std::string>& names) {}
        void operator()(std::vector<double>& state) {}
        
        void operator()() {
          std::cout << std::endl;
        }
        void operator()(const std::string& message) {
          std::cout << message << std::endl;
        }
      };

    }
  }
}

#endif
