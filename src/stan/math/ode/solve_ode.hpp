#ifndef STAN__MATH__SOLVE_ODE_HPP__
#define STAN__MATH__SOLVE_ODE_HPP__

#include <vector>
#include <boost/numeric/odeint.hpp>

namespace stan {
  
  namespace math {
    
    template <typename F, typename T>
    struct ode_system {
      const F& f_;
      const std::vector<T>& theta_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      ode_system(const F& f,
                 const std::vector<T>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int)
        : f_(f), theta_(theta), x_(x), x_int_(x_int) {
      }
      void operator()(const std::vector<T>& y,
                      std::vector<T>& dy_dt,
                      const T t) {
        dy_dt = f(t,y,theta_,x_,x_int_);
      }
    };

    template <typename T>
    struct observer {
      std::vector<std::vector<T> > ys_;
      observer(size_t num_obs_times) {
        ys_.reserve(num_obs_times);
      }
      void operator()(const std::vector<T>& y, const T time) {
        ys_.push_back(y);
      }
    };

    template <typename F, typename T>
    std::vector<std::vector<T> >
    solve_ode(const F f,
              const std::vector<T>& y0,
              const T& t0,
              const std::vector<double>& ts,
              const std::vector<T>& theta,
              const std::vector<double>& x,
              const std::vector<int>& x_int) {
      using namespace boost::numeric::odeint;  // FIXME: trim to what is used

      ode_system<F,T> system(f,theta,x);
      double absolute_tolerance = 1e-6;
      double relative_tolerance = 1e-6;

      std::vector<double> y0_vec(y0.size()*2);
      for (size_t n = 0; n < y0.size(); n++)
        y0_vec[n] = y0[n];
      for (size_t n = y0.size(); n < 2 * y0.size(); n++)
        y0_vec[n] = 0.0;

      std::vector<double> ts_vec(ts.size()+1);
      ts_vec[0] = 0.0;
      for (size_t n = 0; n < ts.size(); n++)
        ts_vec[n+1] = ts[n];
      
      T step_size = ts_vec[1] - ts_vec[0];
      
      observer<T> obs;
      integrate_times(
         make_dense_output(absolute_tolerance,
                           relative_tolerance,
                           runge_kutta_dopri5<std::vector<T>,T,std::vector<T>,T>()),
         ode_system<F,T>(f,theta,x),
         y0_vec,
         boost::begin(ts_vec), boost::end(ts_vec), 
         step_size,
         obs);

      return obs.ys_;
    }
  }


}
#endif
