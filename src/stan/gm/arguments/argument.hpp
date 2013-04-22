#ifndef __STAN__GM__ARGUMENTS__ARGUMENT__HPP__
#define __STAN__GM__ARGUMENTS__ARGUMENT__HPP__

#include <string>
#include <vector>
#include <fstream>

#include <stan/gm/arguments/sub_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class argument {
      
    public:
      
      argument(): _name() {};
      
      ~argument() {
        
        for (std::vector<sub_argument*>::iterator it = _valid_subarguments.begin();
             it != _valid_subarguments.end(); ++it) {
          delete *it;
        }
        
        _valid_subarguments.clear();
        
      }
      
      std::string get_name() { return _name; }
      
      void parse_subargument(std::string s, double v) {
        
        for (std::vector<sub_argument*>::iterator it = _valid_subarguments.begin();
             it != _valid_subarguments.end(); ++it) {
          if (s == (*it)->get_name()) {
            (*it)->set_value(v);
            return;
          }
        }
        
        std::cout << "WARNING: " << s << " is not a valid subcommand for "
                  << _name << std::endl;
        std::cout << "         and will be ignored" << std::endl;
                               
      }
      
      void print_subarguments(std::ostream* s) {
        if(!s) return;
        
        for (std::vector<sub_argument*>::iterator it = _valid_subarguments.begin();
             it != _valid_subarguments.end(); ++it) {
          *s << "\t" << (*it)->get_name() << "\t" << (*it)->get_value();
          if((*it)->is_default()) *s << " (Default)";
          *s << std::endl;
          
        }
      }
      
      virtual void print_help(std::ostream* s) {};
      
    protected:
      
      std::string _name;
      std::vector<sub_argument*> _valid_subarguments;
      
    };
    
  } // gm
  
} // stan

#endif

