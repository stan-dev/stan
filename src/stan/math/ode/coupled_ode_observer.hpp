#ifndef STAN__MATH__ODE__COUPLED_ODE_OBSERVER_HPP
#define STAN__MATH__ODE__COUPLED_ODE_OBSERVER_HPP

#include <vector>
#include <stan/agrad/rev/var.hpp>
 
namespace stan {
  
  namespace math {

    /**
     * Observer for the coupled states.
     */
    struct coupled_ode_observer {
      std::vector<std::vector<double> >& y_coupled_;
      int n;
        
      /**
       * Constructor.
       *
       * @param y_coupled is a reference to a vector of vector of
       *   doubles. The outer vector must have the right number
       *   of elements allocated.
       */
      coupled_ode_observer(std::vector<std::vector<double> >& y_coupled)
        : y_coupled_(y_coupled), n(0) {
      }

      /**
       * operator(). This is what boost's ode solver uses to
       * record values.
       *
       * @param coupled_state the coupled state for the time in t
       * @param t the time
       */
      void operator()(const std::vector<double>& coupled_state, const double t) {
        y_coupled_[n] = coupled_state;
        n++;
      }
    };
    
    
  }
}
#endif
