#ifndef STAN_INTERFACE_CALLBACKS_WRITER_CERR_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_CERR_HPP

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
        
        void operator()(const std::string& key,
                        const double* values,
                        int n_values) {
          if (n_values == 0) return;
          
          std::cerr << key << " = ";
          
          std::cerr << values[0];
          for (int n = 1; n < n_values; ++n)
            std::cerr << "," << values[n];
          std::cerr << std::endl;
        }
        
        void operator()(const std::string& key,
                        const double* values,
                        int n_rows, int n_cols) {
          if (n_rows == 0 || n_cols == 0) return;
          
          std::cerr << key << ":" << std::endl;
          
          for (int i = 0; i < n_rows; ++i) {
            std::cerr << "," << values[i];
            for (int j = 1; j < n_cols; ++j)
              std::cerr << "," << values[i * n_cols + j];
            std::cerr << std::endl;
          }
        }
        
        void operator()(const std::vector<std::string>& names) {}
        void operator()(const std::vector<double>& state) {}
        
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
