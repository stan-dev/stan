#ifndef __STAN__GM__ARGUMENTS__SINGLETON__ARGUMENT__BETA__
#define __STAN__GM__ARGUMENTS__SINGLETON__ARGUMENT__BETA__

#include <stan/gm/arguments/valued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    template<typename T>
    class singleton_argument: public valued_argument {
      
    public:
      
      singleton_argument(): _validity("All") {};
      
      bool parse_args(std::vector<std::string>& args, std::ostream* err) {
        
        if(args.size() == 0) return true;
        
        std::string name;
        std::string value;
        split_arg(args.back(), name, value);
        
        if(_name == name) {
          args.pop_back();
          
          T proposed_value = boost::lexical_cast<T>(value);
          
          if (!set_value(boost::lexical_cast<T>(value))) {

            if(err) {
              *err << proposed_value << " is not a valid value for \"" << _name << "\"" << std::endl;
              *err << std::string(indent_width, ' ') << "Valid values:" << print_valid() << std::endl;
            }
            
            args.clear();
            return false;
            
          }
          
        }
        
        return true;
        
      };
      
      T value() { return _value; }
      
      bool set_value(T value) {
        
        if (is_valid(value)) {
          _value = value;
          return true;
        }
        
        return false;
      
      }

      std::string print_value() { return boost::lexical_cast<std::string>(_value); }
      std::string print_valid() { return " " + _validity; }
      bool is_default() { return _value == _default_value; }
      

    protected:
      
      std::string _validity;
      virtual bool is_valid(T value) { return true; }
      
      T _value;
      T _default_value;
      
    };
    
    typedef singleton_argument<double> real_argument;
    typedef singleton_argument<int> int_argument;
    typedef singleton_argument<bool> bool_argument;
    typedef singleton_argument<std::string> string_argument;
    
  } // gm
  
} // stan

#endif