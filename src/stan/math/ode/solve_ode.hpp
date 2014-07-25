#ifndef STAN__MATH__SOLVE_ODE_HPP__
#define STAN__MATH__SOLVE_ODE_HPP__

#include <vector>
#include <boost/numeric/odeint.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  
  namespace math {
    
    // copy Eigen matrix (or vector or row vector) into std vector, resizing vector
    template <typename T, int R, int C>
    inline
    void to_std_vector(const Eigen::Matrix<T,R,C>& m,
                       std::vector<T>& v) {
      v.assign(&m(0), &m(0) + m.size());
    }

    // return Eigen vector copy of input std vector
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    to_eigen_vector(const std::vector<T>& v) {
      Eigen::Matrix<T,Eigen::Dynamic,1> m(v.size());
      for (int i = 0; i < m.size(); ++i)
        m(i) = v[i];
      return m;
    }
                         

    template <typename F, typename T>
    struct ode_system {
      const F& f_;
      const Eigen::Matrix<T,Eigen::Dynamic,1>& theta_;
      const Eigen::Matrix<double,Eigen::Dynamic,1>& x_;
      ode_system(const F& f,
                 const Eigen::Matrix<T,Eigen::Dynamic,1>& theta,
                 const Eigen::Matrix<double,Eigen::Dynamic,1>& x)
        : f_(f), theta_(theta), x_(x) {
      }
      void operator()(const std::vector<T>& y,
                      std::vector<T>& dy_dt,
                      const T t) {
        to_std_vector(f_(t,to_eigen_vector(y),theta_,x_),
                      dy_dt);
      }
    };

    template <typename T>
    struct observer {
      std::vector<Eigen::Matrix<T,Eigen::Dynamic,1> > ys_;
      observer(size_t num_obs_times) {
        ys_.reserve(num_obs_times);
      }
      void operator()(const std::vector<T>& y, const T time) {
        ys_.push_back(to_eigen_vector(y));
      }
    };

    template <typename F, typename T>
    std::vector<Eigen::Matrix<T,Eigen::Dynamic,1> >
    solve_ode(const F f,
              const Eigen::Matrix<T,Eigen::Dynamic,1>& y0,
              const T& t0,
              const Eigen::Matrix<T,Eigen::Dynamic,1>& ts,
              const Eigen::Matrix<T,Eigen::Dynamic,1>& theta,
              const Eigen::Matrix<double,Eigen::Dynamic,1>& x) {
      using namespace boost::numeric::odeint;  // FIXME: trim to what is used

      ode_system<F,T> system(f,theta,x);
      double absolute_tolerance = 1e-6;
      double relative_tolerance = 1e-6;

      vector<double> y0_vec(y0.size()*2);
      for (size_t n = 0; n < y0.size(); n++)
        y0_vec[n] = y0[n];
      for (size_t n = y0.size(); n < 2*y0.size(); n++)
        y0_vec[n] = 0.0;

      std::vector<double> ts_vec(ts.size()+1);
      ts_vec[0] = 0.0;
      for (size_t n = 0; n < t.ssize(); n++)
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
