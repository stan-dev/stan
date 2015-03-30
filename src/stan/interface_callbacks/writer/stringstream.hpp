#ifndef STAN__INTERFACE_CALLBACKS__WRITER__STRINGSTREAM_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__STRINGSTREAM_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <sstream>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      // An implementation of base_writer
      // using stringstream for unit tests
      
      class stringstream: public base_writer {
      public:
        void operator()(const std::string& key, double value) {
          clear();
          stream_ << key << " = " << value << std::endl;
        }
        void operator()(const std::string& key, const std::string& value) {
          clear();
          stream_ << key << " = " << value << std::endl;
        }
        
        void operator()(const std::string& key,
                        const double* values,
                        int n_values) {
          if (n_values == 0) return;
          clear();
          
          stream_ << key << " = ";
          
          stream_ << values[0];
          for (int n = 1; n < n_values; ++n)
            stream_ << "," << values[n];
          stream_ << std::endl;
        }
        
        void operator()(const std::string& key,
                        const double* values,
                        int n_rows, int n_cols) {
          if (n_rows == 0 || n_cols == 0) return;
          clear();
          
          stream_ << key << ":" << std::endl;
          
          for (int i = 0; i < n_rows; ++i) {
            stream_ << "," << values[i];
            for (int j = 1; j < n_cols; ++j)
              stream_ << "," << values[i * n_cols + j];
            stream_ << std::endl;
          }
        }
        
        void operator()(const std::vector<std::string>& names) {
          if (names.empty()) return;
          clear();
          
          std::vector<std::string>::const_iterator last = names.end();
          --last;
          
          for (std::vector<std::string>::const_iterator it = names.begin(); it != last; ++it)
            stream_ << *it << ",";
          stream_ << names.back() << std::endl;
        }
        
        void operator()(const std::vector<double>& state) {
          if (state.empty()) return;
          clear();
          
          std::vector<double>::const_iterator last = state.end();
          --last;
          
          for (std::vector<double>::const_iterator it = state.begin(); it != last; ++it)
            stream_ << *it << ",";
          stream_ << state.back() << std::endl;
        }
        
        void operator()() {
          clear();
          stream_ << std::endl;
        }
        void operator()(const std::string& message) {
          clear();
          stream_ << message << std::endl;
        }
      };
      
      void clear() {
        stream_.str(std::string());
        stream_.clear();
      }
      
      std::string content() {
        return stream_.str();
      }
      
    private:
      std::stringstream stream_;

    }
  }
}

#endif
