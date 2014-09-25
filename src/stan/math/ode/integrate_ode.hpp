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
#include <stan/math/ode/compute_results.hpp>
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
                  const double& t0,               // initial time
                  const std::vector<double>& ts,  // times at desired solutions
                  const std::vector<T2>& theta,   // parameters
                  const std::vector<double>& x,   // double data values
                  const std::vector<int>& x_int,   // int data values
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
      stan::math::check_nonzero_size("integrate_ode(%1%)",ts,"time_vec", 
                                     static_cast<double*>(0));
      stan::math::check_nonzero_size("integrate_ode(%1%)",y0,"y0_vec",
                                     static_cast<double*>(0));
      stan::math::check_ordered("integrate_ode(%1%)", ts, "times", 
                                static_cast<double*>(0));
      stan::math::check_less("integrate_ode(%1%)",t0,ts[0],"initial time",
                             static_cast<double*>(0));

      
      const int N = y0.size();
      const int M = theta.size();
      // setup y0
      std::vector<double> y0_dbl(N);
      for (int n = 0; n < N; n++)
        y0_dbl[n] = value_of(y0[n]);

      // setup theta
      std::vector<double> theta_dbl(M);
      for (int m = 0; m < M; m++)
        theta_dbl[m] = value_of(theta[m]);

      // builds coupled ode system
      coupled_ode_system<F, T1, T2> system(f, y0_dbl, theta_dbl, x, x_int, N, pstream);

      // set up the coupled state. base system has size N.
      // y0,     theta,  size
      // double, double, N
      // double, var,    N + N * M
      // var,    double, N + N * N
      // var,    var,    N + N * (M + N)
      int coupled_state_size 
        = N + N * (is_same<var,T1>::value * N
                   + is_same<var,T2>::value * M);
      std::vector<double> coupled_state(coupled_state_size, 0.0);
      
      // set the initial coupled_state values to y0 if 
      // we don't need the sensitivities of the y0.
      if (is_same<double, T1>::value)
        for (int n = 0; n < N; n++)
          coupled_state[n] = value_of(y0[n]);
      
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
                      system,
                      coupled_state, 
                      boost::begin(ts_vec), boost::end(ts_vec), 
                      step_size,
                      observer);

      // remove the state corresponding to the initial value
      y_coupled.erase(y_coupled.begin());

      std::vector<std::vector<typename stan::return_type<T1,T2>::type> > 
        y_vec = compute_results(y_coupled, y0, theta);

      return y_vec;
    }
                   
  }
}
#endif
