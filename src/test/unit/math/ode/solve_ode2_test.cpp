#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev.hpp>

#include <stan/math/ode/solve_ode.hpp>
#include <stan/math/ode/solve_ode2.hpp>

template <typename T>
inline
std::vector<T> harm_osc_ode(const T& t_in, // initial time
                            const std::vector<T>& y_in, //initial positions
                            const std::vector<T>& theta, // parameters
                            const std::vector<double>& x, // double data
                            const std::vector<int>& x_int) { // integer data
  std::vector<T> res;
  res.push_back(y_in[1]);
  res.push_back(-y_in[0] - theta[0]*y_in[1]);

  return res;
}

struct harm_osc_ode_fun {
  template <typename T>
  inline 
  std::vector<T> operator()(const T& t_in, // initial time
                            const std::vector<T>& y_in, //initial positions
                            const std::vector<T>& theta, // parameters
                            const std::vector<double>& x, // double data
                            const std::vector<int>& x_int) const { // integer data
    return harm_osc_ode(t_in, y_in, theta, x, x_int);
  }
};

TEST(solve_ode2, ode_system2) {
  using stan::math::ode_system2;

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

  ode_system2<harm_osc_ode_fun> system(harm_osc, theta, x, x_int);

  system(y0, dy_dt, t0);

  EXPECT_FLOAT_EQ(0.5, dy_dt[0]);
  EXPECT_FLOAT_EQ(-1.075, dy_dt[1]);
  EXPECT_FLOAT_EQ(2, dy_dt[2]);
  EXPECT_FLOAT_EQ(-1.8, dy_dt[3]);
}

template <typename F>
inline void solve_ode_efficient(const F& f,
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

  stan::agrad::var t0;
  t0 = t0_dbl;

  std::vector<stan::agrad::var> ts;
  for (int i = 0; i < ts_dbl.size(); i++)
    ts.push_back(ts_dbl[i]);

  std::vector<stan::agrad::var> theta;
  for (int i = 0; i < theta_dbl.size(); i++)
    theta.push_back(theta_dbl[i]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::solve_ode2(f, y0, t0,
                                   ts, theta, x, x_int);
  value = ode_res[iteration_number][eqn_number].val();
  
  ode_res[iteration_number][eqn_number].grad(theta, gradients);
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

  stan::agrad::var t0;
  t0 = t0_dbl;

  std::vector<stan::agrad::var> ts;
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

TEST(solve_ode2, harm_osc_compare_to_diff_integrator) {
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

  for (int i = 0; i < ts.size()+1; i++) {
    for (int j = 0; j < theta.size(); j++) {
      double val_diff_integrator;
      std::vector<double> grad_diff_integrator;
      double val_eff;
      std::vector<double> grad_eff;
      solve_ode_diff_integrator(harm_osc, y0, t0, ts, theta, x, x_int, i, j, 
                                val_diff_integrator, grad_diff_integrator);
      solve_ode_efficient(harm_osc, y0, t0, ts, theta, x, x_int, i, j, 
                          val_eff, grad_eff);
      EXPECT_NEAR(val_diff_integrator, val_eff, 1e-5);
      
      for (int k = 0; k < theta.size(); k++)
        EXPECT_NEAR(grad_diff_integrator[k], grad_eff[k], 1e-5);

      
    }
  }
}

TEST(solve_ode2, harm_osc) {
  harm_osc_ode_fun harm_osc;

  std::vector<stan::agrad::var> y0;
  std::vector<stan::agrad::var> theta;
  stan::agrad::var t0;
  std::vector<std::vector<stan::agrad::var> > ode_res;
  std::vector<stan::agrad::var> ts;

  stan::agrad::var gamma(0.15);
  t0 = 0;

  theta.push_back(gamma);
  y0.push_back(1.0);
  y0.push_back(0.0);

  std::vector<double> x;
  std::vector<int> x_int;

  for (int i = 0; i < 100; i++)
    ts.push_back(0.1*(i+1));

  ode_res = stan::math::solve_ode2(harm_osc, y0, t0,
                                  ts, theta, x, x_int);

  EXPECT_NEAR(0.995029, ode_res[1][0].val(), 1e-5);
  EXPECT_NEAR(-0.0990884, ode_res[1][1].val(), 1e-5);

  EXPECT_NEAR(-0.421907, ode_res[100][0].val(), 1e-5);
  EXPECT_NEAR(0.246407, ode_res[100][1].val(), 1e-5);

}
