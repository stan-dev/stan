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

#include <stan/math/ode/ode_system.hpp>
#include <stan/math/ode/compute_results.hpp>
#include <stan/math/ode/push_back_state_and_time.hpp>

namespace stan {
  
  namespace math {

    template <typename F, typename T1, typename T2>
    std::vector<std::vector<typename stan::return_type<T1,T2>::type> >
    integrate_ode(const F& f,
                  const std::vector<T1> y0, 
                  const double& t0, // initial time
                  const std::vector<double>& ts, // times at desired solutions
                  const std::vector<T2>& theta, // parameters
                  const std::vector<double>& x, // double data values
                  const std::vector<int>& x_int, // int data values
                  std::ostream* pstream) { 
      using boost::numeric::odeint::integrate_times;  
      using boost::numeric::odeint::make_dense_output;  
      using boost::numeric::odeint::runge_kutta_dopri5;
      
      stan::math::check_nonzero_size("integrate_ode(%1%)",ts,"time_vec",
                                     static_cast<double*>(0));
      stan::math::check_nonzero_size("integrate_ode(%1%)",y0,"y0_vec",
                                     static_cast<double*>(0));
      stan::math::check_ordered("integrate_ode(%1%)", ts, "times", 
                                static_cast<double*>(0));
      stan::math::check_less("integrate_ode(%1%)",t0,ts[0],"initial time",
                             static_cast<double*>(0));

      double absolute_tolerance = 1e-6;
      double relative_tolerance = 1e-6;

      std::vector<double> theta_dbl;
      for (int i = 0; i < theta.size(); i++)
        theta_dbl.push_back(value_of(theta[i]));

      std::vector<double> y0_vec; 
      for (size_t n = 0; n < y0.size(); n++)
        y0_vec.push_back(value_of(y0[n]));

      //initialize values to 0 if theta is var
      if (boost::is_same<stan::agrad::var, T2>::value)
        for (size_t n = 0; n < theta.size() * y0.size(); n++)
          y0_vec.push_back(0.0);

      //initalize values to 0 if y0 is var
      if (boost::is_same<stan::agrad::var, T1>::value)
        for (size_t n = 0; n < y0.size() * y0.size(); n++)
          y0_vec.push_back(0.0);

      // builds coupled ode system
      ode_system<F, T1, T2> system(f,y0_vec,theta_dbl,x,x_int, y0.size(),pstream);

      // sets initial positions to 0 if y0 is var
      if (boost::is_same<stan::agrad::var, T1>::value)
        for (size_t n = 0; n < y0.size(); n++)
          y0_vec[n] = 0;

      std::vector<double> ts_vec(ts.size()+1);
      ts_vec[0] = t0;
      for (size_t n = 0; n < ts.size(); n++)
        ts_vec[n+1] = ts[n];
      
      double step_size = 0.1;

      std::vector<std::vector<double> > x_vec;
      std::vector<double> t_vec;
      push_back_state_and_time<double> obs(x_vec, t_vec);
      integrate_times(make_dense_output(absolute_tolerance,
                                        relative_tolerance,
                                        runge_kutta_dopri5<std::vector<double>,
                                        double,std::vector<double>,double>()),
                      system,
                      y0_vec,
                      boost::begin(ts_vec), boost::end(ts_vec), 
                      step_size,
                      obs);

      std::vector<std::vector<double> > y = obs.get();
      std::vector<std::vector<typename stan::return_type<T1,T2>::type> > 
        res = compute_results(y, y0, theta);

      res.erase(res.begin());

      // add back initial positions if y0 is var
      if (boost::is_same<stan::agrad::var, T1>::value)
        for (size_t n = 0; n < res.size(); n++)
          for (size_t m = 0; m < y0.size(); m++)
            res[n][m] += y0[m];

      return res;
    }
                   
  }
}
#endif
