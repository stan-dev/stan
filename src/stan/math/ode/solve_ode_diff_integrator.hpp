// #ifndef STAN__MATH__ODE__SOLVE_ODE_DIFF_INTEGRATOR_HPP__
// #define STAN__MATH__ODE__SOLVE_ODE_DIFF_INTEGRATOR_HPP__

// #include <vector>
// #include <boost/numeric/odeint.hpp>
// #include <stan/agrad/rev/var.hpp>
// #include <stan/math/ode/util.hpp>

// namespace stan {
  
//   namespace math {

//     template <typename F, typename T>
//     struct ode_system_diff_integrator {
//       const F& f_;
//       const std::vector<T>& theta_;
//       const std::vector<double>& x_;
//       const std::vector<int>& x_int_;
//       ode_system_diff_integrator(const F& f,
//                                  const std::vector<T>& theta,
//                                  const std::vector<double>& x,
//                                  const std::vector<int>& x_int)
//         : f_(f), theta_(theta), x_(x), x_int_(x_int) {
//       }
//       void operator()(const std::vector<T>& y,
//                       std::vector<T>& dy_dt,
//                       const T& t) {
//         dy_dt = f_(t,y,theta_,x_,x_int_);
//       }
//     };

//     template <typename F, typename T>
//     std::vector<std::vector<T> >
//     integrate_ode_diff_integrator(const F& f,
//                               const std::vector<T>& y0,
//                               const double& t0,
//                               const std::vector<double>& ts,
//                               const std::vector<T>& theta,
//                               const std::vector<double>& x,
//                               const std::vector<int>& x_int) {
//       using namespace boost::numeric::odeint;  // FIXME: trim to what is used
//       using namespace std;

//       ode_system_diff_integrator<F,T> system(f,theta,x,x_int);
//       double absolute_tolerance = 1e-6;
//       double relative_tolerance = 1e-6;

//       std::vector<T> y0_vec(y0.size());
//       for (size_t n = 0; n < y0.size(); n++)
//         y0_vec[n] = y0[n];

//       std::vector<T> ts_vec(ts.size()+1);
//       ts_vec[0] = t0;
//       for (size_t n = 0; n < ts.size(); n++)
//         ts_vec[n+1] = ts[n];
      
//       T step_size = 0.1;
  
//       std::vector<std::vector<T> > x_vec;
//       std::vector<T> t_vec;
//       push_back_state_and_time<T> obs(x_vec, t_vec);

//       integrate_times(
//          make_dense_output(absolute_tolerance,
//                            relative_tolerance,
//                            runge_kutta_dopri5<std::vector<T>,T,std::vector<T>,T>()),
//          system,
//          y0_vec,
//          boost::begin(ts_vec), boost::end(ts_vec), 
//          step_size,
//          obs);

//       std::vector<std::vector<T> > res = obs.get();
//       res.erase(res.begin());
//       return res;
//     }
//   }


// }
// #endif
