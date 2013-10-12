#ifndef __STAN__GM__ARGUMENTS__BFGS__HPP__
#define __STAN__GM__ARGUMENTS__BFGS__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_init_alpha.hpp>
#include <stan/gm/arguments/arg_tolerance.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_bfgs: public categorical_argument {
      
    public:
      
      arg_bfgs() {
        
        _name = "bfgs";
        _description = "BFGS with linesearch";
        
        _subarguments.push_back(new arg_init_alpha());
        _subarguments.push_back(new arg_tolerance("tol_obj","Convergence tolerance on changes in objective function value","1e-8",1e-8));
        _subarguments.push_back(new arg_tolerance("tol_grad","Convergence tolerance on the norm of the gradient","1e-8",1e-8));
        _subarguments.push_back(new arg_tolerance("tol_param","Convergence tolerance on changes in parameter value","1e-8",1e-8));
        
        
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

