#ifndef STAN__INTERFACE_CALLBACKS__WRITER__WRITER_HPP
#define STAN__INTERFACE_CALLBACKS__WRITER__WRITER_HPP

#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>

namespace stan {
  namespace interface_callbacks {
    namespace writer {
      
      class writer {
      public:
        virtual ~writer() {};
        
        virtual void write_key_value(const std::string& key,
                                    double value) = 0;
        virtual void write_key_value(const std::string& key,
                                    const std::string& value) = 0;
        
        virtual void write_state_names(std::vector<std::string>& names) = 0;
        virtual void write_state(std::vector<double>& state) = 0;
        
        virtual void write_message(const std::string& message) = 0;
        
        // FIXME: Replace with std::to_string when we update to C++11
        template <typename T>
        static std::string to_string(T x) {
          return boost::lexical_cast<std::string>(x);
        }
        
        template <typename T>
        static std::string to_string(T x, int width) {
          std::string str = boost::lexical_cast<std::string>(x);
          if (str.size() < width) {
            str.insert(str.begin(), width - str.size(), ' ');
            return str;
          } else
            return str;
        }
        
        static std::string pad(std::string str, int width) {
          if (str.size() < width) {
            str.insert(str.begin(), width - str.size(), ' ');
            return str;
          } else
            return str;
        }
        
      };

    }
  }
}

#endif
