#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev.hpp>

#include <stan/math/ode/solve_ode_diff_integrator.hpp>
#include <stan/math/ode/solve_ode.hpp>

template <typename T0, typename T1, typename T2>
inline
std::vector<typename stan::return_type<T1,T2>::type> 
harm_osc_ode(const T0& t_in, // initial time
             const std::vector<T1>& y_in, //initial positions
             const std::vector<T2>& theta, // parameters
             const std::vector<double>& x, // double data
             const std::vector<int>& x_int) { // integer data
  std::vector<typename stan::return_type<T1,T2>::type> res;
  res.push_back(y_in[1]);
  res.push_back(-y_in[0] - theta[0]*y_in[1]);

  return res;
}

struct harm_osc_ode_fun {
  template <typename T0, typename T1, typename T2>
  inline 
  std::vector<typename stan::return_type<T1,T2>::type> 
  operator()(const T0& t_in, // initial time
             const std::vector<T1>& y_in, //initial positions
             const std::vector<T2>& theta, // parameters
             const std::vector<double>& x, // double data
             const std::vector<int>& x_int) const { // integer data
    return harm_osc_ode(t_in, y_in, theta, x, x_int);
  }
};

TEST(solve_ode, ode_system) {
  using stan::math::ode_system;

  harm_osc_ode_fun harm_osc;

  std::vector<double> theta;
  std::vector<double> y0;
  double t0;
  std::vector<double> dy_dt;

  double gamma(0.15);
  t0 = 0;

  theta.push_back(gamma);
  y0.push_back(1.0);
  y0.push_back(0.5);
  y0.push_back(1.0);
  y0.push_back(2.0);

  std::vector<double> x;
  std::vector<int> x_int;

  ode_system<harm_osc_ode_fun, double, stan::agrad::var> system(harm_osc, y0,theta, x, x_int);

  system(y0, dy_dt, t0);

  EXPECT_FLOAT_EQ(0.5, dy_dt[0]);
  EXPECT_FLOAT_EQ(-1.075, dy_dt[1]);
  EXPECT_FLOAT_EQ(2, dy_dt[2]);
  EXPECT_FLOAT_EQ(-1.8, dy_dt[3]);
}

template <typename F>
inline void solve_ode_efficient_double_var(const F& f,
                                           const std::vector<double>& y0_dbl,
                                           const double& t0_dbl,
                                           const std::vector<double>& ts_dbl,
                                           const std::vector<double>& theta_dbl,
                                           const std::vector<double>& x,
                                           const std::vector<int>& x_int,
                                           const int& iteration_number,
                                           const int& eqn_number,
                                           double& value,
                                           std::vector<double>& gradients) {

  std::vector<double> y0;
  for (int i = 0; i < y0_dbl.size(); i++)
    y0.push_back(y0_dbl[i]);

  double t0;
  t0 = t0_dbl;

  std::vector<double> ts;
  for (int i = 0; i < ts_dbl.size(); i++)
    ts.push_back(ts_dbl[i]);

  std::vector<stan::agrad::var> theta;
  for (int i = 0; i < theta_dbl.size(); i++)
    theta.push_back(theta_dbl[i]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::solve_ode(f, y0, t0,
                                   ts, theta, x, x_int);
  value = ode_res[iteration_number][eqn_number].val();
  
  ode_res[iteration_number][eqn_number].grad(theta, gradients);
}

template <typename F>
inline void solve_ode_efficient_var_double(const F& f,
                                           const std::vector<double>& y0_dbl,
                                           const double& t0_dbl,
                                           const std::vector<double>& ts_dbl,
                                           const std::vector<double>& theta_dbl,
                                           const std::vector<double>& x,
                                           const std::vector<int>& x_int,
                                           const int& iteration_number,
                                           const int& eqn_number,
                                           double& value,
                                           std::vector<double>& gradients) {

  std::vector<stan::agrad::var> y0;
  for (int i = 0; i < y0_dbl.size(); i++)
    y0.push_back(y0_dbl[i]);

  double t0;
  t0 = t0_dbl;

  std::vector<double> ts;
  for (int i = 0; i < ts_dbl.size(); i++)
    ts.push_back(ts_dbl[i]);

  std::vector<double> theta;
  for (int i = 0; i < theta_dbl.size(); i++)
    theta.push_back(theta_dbl[i]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::solve_ode(f, y0, t0,
                                   ts, theta, x, x_int);
  value = ode_res[iteration_number][eqn_number].val();
  
  ode_res[iteration_number][eqn_number].grad(y0, gradients);
}

template <typename F>
inline void solve_ode_diff_integrator(const F& f,
                                      const std::vector<double>& y0_dbl,
                                      const double& t0_dbl,
                                      const std::vector<double>& ts_dbl,
                                      const std::vector<double>& theta_dbl,
                                      const std::vector<double>& x,
                                      const std::vector<int>& x_int,
                                      const int& iteration_number,
                                      const int& eqn_number,
                                      double& value,
                                      std::vector<double>& gradients) {

  std::vector<stan::agrad::var> y0;
  for (int i = 0; i < y0_dbl.size(); i++)
    y0.push_back(y0_dbl[i]);

  double t0;
  t0 = t0_dbl;

  std::vector<double> ts;
  for (int i = 0; i < ts_dbl.size(); i++)
    ts.push_back(ts_dbl[i]);

  std::vector<stan::agrad::var> theta;
  for (int i = 0; i < theta_dbl.size(); i++)
    theta.push_back(theta_dbl[i]);
  
  std::vector<stan::agrad::var> vars;
  for (int i = 0; i < theta.size(); i++)
    vars.push_back(theta[i]);

  for (int i = 0; i < y0.size(); i++)
    vars.push_back(y0[i]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::solve_ode_diff_integrator(f, y0, t0,
                                  ts, theta, x, x_int);
  value = ode_res[iteration_number][eqn_number].val();
  
  ode_res[iteration_number][eqn_number].grad(vars, gradients);
}

TEST(solve_ode, harm_osc_compare_to_diff_integrator_double_var) {
  harm_osc_ode_fun harm_osc;

  std::vector<double> y0;
  std::vector<double> theta;
  double t0;
  std::vector<double> ts;

  t0 = 0;

  theta.push_back(0.15);
  y0.push_back(1.0);
  y0.push_back(0.0);

  std::vector<double> x;
  std::vector<int> x_int;

  for (int i = 0; i < 100; i++)
    ts.push_back(0.1*(i+1));

  for (int i = 1; i < ts.size(); i++) {
    for (int j = 0; j < y0.size(); j++) {
      double val_diff_integrator;
      std::vector<double> grad_diff_integrator;
      double val_eff;
      std::vector<double> grad_eff;
      solve_ode_diff_integrator(harm_osc, y0, t0, ts, theta, x, x_int, i, j, 
                                val_diff_integrator, grad_diff_integrator);
      solve_ode_efficient_double_var(harm_osc, y0, t0, ts, theta, x, x_int, i, j, 
                                     val_eff, grad_eff);
      EXPECT_NEAR(val_diff_integrator, val_eff, 1e-5);
      
      for (int k = 0; k < theta.size(); k++)
        EXPECT_NEAR(grad_diff_integrator[k], grad_eff[k], 1e-5);
    }
  }
}


// TEST(solve_ode, harm_osc_compare_to_diff_integrator_var_double) {
//   harm_osc_ode_fun harm_osc;

//   std::vector<double> y0;
//   std::vector<double> theta;
//   double t0;
//   std::vector<double> ts;

//   t0 = 0;

//   theta.push_back(0.15);
//   y0.push_back(1.0);
//   y0.push_back(0.0);

//   std::vector<double> x;
//   std::vector<int> x_int;

//   for (int i = 0; i < 100; i++)
//     ts.push_back(0.1*(i+1));

//   for (int i = 1; i < ts.size(); i++) {
//     for (int j = 0; j < y0.size(); j++) {
//       double val_diff_integrator;
//       std::vector<double> grad_diff_integrator;
//       double val_eff;
//       std::vector<double> grad_eff;
//       solve_ode_diff_integrator(harm_osc, y0, t0, ts, theta, x, x_int, i, j, 
//                                 val_diff_integrator, grad_diff_integrator);
//       solve_ode_efficient_var_double(harm_osc, y0, t0, ts, theta, x, x_int, i, j, 
//                                      val_eff, grad_eff);
//       EXPECT_NEAR(val_diff_integrator, val_eff, 1e-5);
      
//       for (int k = 0; k < y0.size(); k++)
//         EXPECT_NEAR(grad_diff_integrator[theta.size()+k], grad_eff[k], 1e-5);
//     }
//   }
// }

TEST(solve_ode, harm_osc) {
  harm_osc_ode_fun harm_osc;

  std::vector<stan::agrad::var> y0;
  std::vector<double> theta;
  double t0;
  std::vector<std::vector<stan::agrad::var> > ode_res;
  std::vector<double> ts;

  double gamma(0.15);
  t0 = 0;

  theta.push_back(gamma);
  y0.push_back(1.0);
  y0.push_back(0.0);

  std::vector<double> x;
  std::vector<int> x_int;

  for (int i = 0; i < 100; i++)
    ts.push_back(0.1*(i+1));

  ode_res = stan::math::solve_ode(harm_osc, y0, t0,
                                  ts, theta, x, x_int);

  EXPECT_NEAR(0.995029, ode_res[0][0].val(), 1e-5);
  EXPECT_NEAR(-0.0990884, ode_res[0][1].val(), 1e-5);

  EXPECT_NEAR(-0.421907, ode_res[99][0].val(), 1e-5);
  EXPECT_NEAR(0.246407, ode_res[99][1].val(), 1e-5);
  
  std::vector<double> grads;
  ode_res[99][1].grad(y0, grads);
}
