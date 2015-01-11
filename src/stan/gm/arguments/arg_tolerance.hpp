#ifndef STAN__GM__ARGUMENTS__TOLERANCE_HPP
#define STAN__GM__ARGUMENTS__TOLERANCE_HPP

#include <boost/lexical_cast.hpp>

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_tolerance : public real_argument {
      
    public:
      
      arg_tolerance(const char *name, const char *desc, double def) : real_argument() {
        _name = name;
        _description = desc;
        _validity = "0 <= tol";
        _default = boost::lexical_cast<std::string>(def);
        _default_value = def;
        _constrained = true;
        _good_value = 1.0;
        _bad_value = -1.0;
        _value = _default_value;
      };
      
      bool is_valid(double value) { return value >= 0; }

    };
    
  } // gm
  
} // stan

#endif
