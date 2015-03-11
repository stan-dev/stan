#ifndef STAN__INTERFACE_CALLBACKS__WRITER__WRITER_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__WRITER_HPP

#include <string>
#include <vector>
#include <ostringstream>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      class writer {
      public:
        virtual ~writer() {};
        
        virtual void writer_key_value(const std::string& key,
                                      double value) = 0;
        virtual void writer_key_value(const std::string& key,
                                      const std::string& value) = 0;
        
        virtual void write_state_names(const std::vector<std::string>& names) = 0;
        virtual void write_state(const std::vector<double>& state) = 0;
        
        virtual void write_message(const std::string& message) = 0;
        
        // FIXME: Replace with std::to_string when we update to C++11
        static to_string(double x) {
          static_cast<std::ostringstream*>( &(std::ostringstream() << x) )->str();
        }
        
        static to_string(double x, int width) {
          static_cast<std::ostringstream*>( &(std::ostringstream()
                                              << std::setw(width) << x) )->str();
        }
        
        static pad(std::string& str, int width) {
          if (str.size() < width)
            str.insert(str.begin(), width - str.size(), ' ');
        }
        
      };

    }
  }
}

#endif
