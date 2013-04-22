#ifndef __STAN__GM__ARGUMENTS__SUB__ARGUMENT__HPP__
#define __STAN__GM__ARGUMENTS__SUB__ARGUMENT__HPP__

#include <fstream>
#include <string>

namespace stan {
  
  namespace gm {
    
    class sub_argument {
      
    public:
      
      sub_argument(): _name(), _value(0.0), _default(true) {};
      virtual ~sub_argument() {};
      
      std::string get_name() { return _name; }
      double get_value() { return _value; }
      bool is_default() { return _default; }
      
      virtual bool valid_value(double v) {
        return true;
      }
      
      void set_value(double v) {
        if (valid_value(v)) {
          _value = v;
          _default = false;
        }
      }
      
      virtual void print_help(std::ostream* s) {};
      
    protected:
      
      std::string _name;
      double _value;
      bool _default;
      
    };
    
  } // gm
  
} // stan

#endif

