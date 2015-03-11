#ifndef STAN__INTERFACE_CALLBACKS__WRITER__FSTREAM_CSV_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__FSTREAM_CSV_HPP

#include <stan/interface_callbacks/writer/writer.hpp>
#include <string>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      // FIXME: Move to CmdStan
      class fstream_csv: public writer {
      public:
        fstream_csv(const std::string& filename):
          output(filename.c_str(), std::fstream::out) {}
        
        bool good() { return output.good(); }
        
        void writer_key_value(const std::string& key, double value) {
          output << key << " = " << value << std::endl;
        };
        
        void writer_key_value(const std::string& key, const std::string& value) {
          output << key << " = " << value << std::endl;
        };
        
        void write_state_names(const std::vector<std::string>& names) {
          if (names.empty()) return;
          
          std::vector<std::string>::iterator last = names.back();
          --last;
          
          for (std::vector<std::string>::iterator it = names.begin(); it != last; ++it)
            output << *it << ",";
          output << names.last() << std::endl;
        };
        
        void write_state(const std::vector<double>& state) {
          if (state.empty()) return;
          
          std::vector<double>::iterator last = state.back();
          --last;
          
          for (std::vector<double>::iterator it = state.begin(); it != last; ++it)
            output << *it << ",";
          output << state.last() << std::endl;
        };
        
        void write_message(const std::string& message) {
          output << message << std::endl;
        };
        
      private:
        std::fstream output;
      };

    }
  }
}

#endif
