#ifndef STAN__MATH__ODE__INTEGRATE_ODE_HPP
#define STAN__MATH__ODE__INTEGRATE_ODE_HPP

#include <ostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/math/error_handling/check_less.hpp>
#include <stan/math/error_handling/matrix/check_ordered.hpp>

#include <stan/math/ode/coupled_ode_system.hpp>
#include <stan/math/ode/coupled_ode_observer.hpp>

namespace stan {
  
  namespace math {
    
    /**
     * integrate_ode numerically solves the ordinary differential
     * equation specified for the times provided.
     *
     * This function is templated to allow the initial times to be
     * either data or autodiff variables and the parameters to be data
     * or autodiff variables.
     *
     * This version of integrate_ode uses boost odeint's
     * runge_kutta_dopri5 solver.
     *
     * @tparam F ode system function concept
     * @tparam T1 type of the initial values
     * @tparam T2 type of the parameters
     * 
     * @param[in] f a functor for the base ordinary differential equation.
     * @param[in] y0 the initial state. The size of the initial state must
     *    be greater than 0.
     * @param[in] t0 the time of the initial state.
     * @param[in] ts the times of the desired solutions.
     * @param[in] theta the parameters of the ode
     * @param[in] x double data values that can be used by the ode
     * @param[in] x_int integer data values that can be used by the ode
     * @param[in,out] pstream the print stream for messages
     *
     * @returns a vector of states, each state corresponding to a time
     *   in ts.
     */
    template <typename F, typename T1, typename T2>
    std::vector<std::vector<typename stan::return_type<T1,T2>::type> >
    integrate_ode(const F& f,
                  const std::vector<T1> y0, 
                  const double t0,
                  const std::vector<double>& ts,
                  const std::vector<T2>& theta,
                  const std::vector<double>& x,
                  const std::vector<int>& x_int,
                  std::ostream* pstream) {            
      using boost::numeric::odeint::integrate_times;  
      using boost::numeric::odeint::make_dense_output;  
      using boost::numeric::odeint::runge_kutta_dopri5;
      using boost::is_same;
      using stan::agrad::var;
      
      const double absolute_tolerance = 1e-6;
      const double relative_tolerance = 1e-6;
      const double step_size = 0.1;

      // validate inputs
      stan::math::check_nonzero_size("integrate_ode(%1%)", ts, "time", 
                                     static_cast<double*>(0));
      stan::math::check_nonzero_size("integrate_ode(%1%)", y0, "initial state",
                                     static_cast<double*>(0));
      stan::math::check_ordered("integrate_ode(%1%)", ts, "times", 
                                static_cast<double*>(0));
      stan::math::check_less("integrate_ode(%1%)",t0, ts[0], "initial time",
                             static_cast<double*>(0));

      coupled_ode_system<F, T1, T2>
        coupled_system(f, y0, theta, x, x_int, pstream);
      std::vector<double> coupled_state = coupled_system.initial_state();
      
      // boost expects the first time in the vector to be the 
      // time of the initial state
      std::vector<double> ts_vec(ts.size()+1);
      ts_vec[0] = t0;
      for (size_t n = 0; n < ts.size(); n++)
        ts_vec[n+1] = ts[n];
      
      std::vector<std::vector<double> > y_coupled(ts_vec.size());
      coupled_ode_observer observer(y_coupled);

      integrate_times(make_dense_output(absolute_tolerance,
                                        relative_tolerance,
                                        runge_kutta_dopri5<std::vector<double>,
                                        double,std::vector<double>,double>()),
                      coupled_system,
                      coupled_state, 
                      boost::begin(ts_vec), boost::end(ts_vec), 
                      step_size,
                      observer);

      // remove the first state; this state corresponds to the initial value
      y_coupled.erase(y_coupled.begin());

      std::vector<std::vector<typename stan::return_type<T1,T2>::type> >
        y_vec = coupled_system.compute_results(y_coupled, y0, theta);

      return y_vec;
    }
                   
  }
}
#endif
