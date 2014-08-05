#ifndef STAN__GM__ARGUMENTS__RESUME__HPP
#define STAN__GM__ARGUMENTS__RESUME__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_resume_load_from_file.hpp>
#include <stan/gm/arguments/arg_resume_save_to_file.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_resume: public categorical_argument {
      
    public:
      
      arg_resume(): categorical_argument() {
        
        _name = "resume";
        _description = "Resume sampling options";
        
        _subarguments.push_back(new arg_resume_load_from_file());
        _subarguments.push_back(new arg_resume_save_to_file());
        
      };
      
    };
    
  } // gm
  
} // stan

#endif
