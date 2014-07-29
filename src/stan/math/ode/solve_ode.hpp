#ifndef STAN__MATH__SOLVE_ODE_HPP__
#define STAN__MATH__SOLVE_ODE_HPP__

#include <vector>
#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev/var.hpp>
 
namespace stan {
  namespace agrad {
    stan::agrad::var max(stan::agrad::var a, stan::agrad::var b) {
      return fmax(a, b);
    }
  }
}

namespace stan {
  
  namespace math {

    template <typename F, typename T>
    struct ode_system {
      F f_;
      const std::vector<T>& theta_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      ode_system(F f,
                 const std::vector<T>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int)
        : f_(f), theta_(theta), x_(x), x_int_(x_int) {
      }
      void operator()(const std::vector<T>& y,
                      std::vector<T>& dy_dt,
                      const T& t) {
        dy_dt = f_(t,y,theta_,x_,x_int_);
      }
    };

    template<class T>
    struct push_back_state_and_time {
      std::vector< std::vector<T> >& m_states;
      std::vector< T >& m_times;
      
      push_back_state_and_time(std::vector< std::vector<T> > &states,
                               std::vector< T > &times)
        : m_states( states ), 
          m_times( times ) { }
      
      void operator()(const std::vector<T> &x, T t) {
        m_states.push_back( x );
        m_times.push_back( t );
      }
      
      std::vector<std::vector<T> > get() {
        return m_states;
      }
      
      void print() {
        std::cout << "time,x_0";
        for (size_t n = 1; n < m_states[0].size(); n++)
          std::cout << ",x_" << n;
        std::cout << std::endl;
        for (size_t n = 0; n < m_states.size(); n++) {
          std::cout << m_times[n]
                    << "," << m_states[n][0]
                    << "," << m_states[n][1]
                    << std::endl;
        }
      }
    };
    

    template <typename F, typename T>
    std::vector<std::vector<T> >
    solve_ode(F f,
              const std::vector<T>& y0,
              const T& t0,
              const std::vector<T>& ts,
              const std::vector<T>& theta,
              const std::vector<double>& x,
              const std::vector<int>& x_int) {
      using namespace boost::numeric::odeint;  // FIXME: trim to what is used
      using namespace std;

      ode_system<F,T> system(f,theta,x,x_int);
      double absolute_tolerance = 1e-6;
      double relative_tolerance = 1e-6;

      std::vector<T> y0_vec(y0.size());
      for (size_t n = 0; n < y0.size(); n++)
        y0_vec[n] = y0[n];
      // for (size_t n = y0.size(); n < 2 * y0.size(); n++)
      //   y0_vec[n] = 0.0;

      std::vector<T> ts_vec(ts.size()+1);
      ts_vec[0] = 0.0;
      for (size_t n = 0; n < ts.size(); n++)
        ts_vec[n+1] = ts[n];
      
      T step_size = ts_vec[1] - ts_vec[0];
  
      std::vector<std::vector<T> > x_vec;
      std::vector<T> t_vec;
      push_back_state_and_time<T> obs(x_vec, t_vec);

      integrate_times(
         make_dense_output(absolute_tolerance,
                           relative_tolerance,
                           runge_kutta_dopri5<std::vector<T>,T,std::vector<T>,T>()),
         system,
         y0_vec,
         boost::begin(ts_vec), boost::end(ts_vec), 
         step_size,
         obs);

      return obs.get();
    }
  }


}
#endif
