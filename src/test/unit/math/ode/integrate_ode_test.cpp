#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev.hpp>

#include <stan/math/ode/integrate_ode.hpp>

#include <test/unit/math/ode/util.hpp>

#include <test/unit/math/ode/harmonic_oscillator.hpp>


TEST(integrate_ode, harm_osc_finite_diff) {
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

  test_ode(harm_osc, t0, ts, y0, theta, x, x_int, 1e-8,1e-4);
}

TEST(integrate_ode, harm_osc_known_values_dd) {
  std::stringstream msgs;

  harm_osc_ode_fun harm_osc;

  std::vector<double> y0;
  std::vector<double> theta;
  double t0;
  std::vector<std::vector<double> > ode_res;
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

  ode_res = stan::math::integrate_ode(harm_osc, y0, t0,
                                      ts, theta, x, x_int, &msgs);

  EXPECT_NEAR(0.995029, ode_res[0][0], 1e-5);
  EXPECT_NEAR(-0.0990884, ode_res[0][1], 1e-5);

  EXPECT_NEAR(-0.421907, ode_res[99][0], 1e-5);
  EXPECT_NEAR(0.246407, ode_res[99][1], 1e-5);
}

TEST(integrate_ode, harm_osc_known_values_dv) {
  std::stringstream msgs;

  harm_osc_ode_fun harm_osc;

  std::vector<double> y0;
  std::vector<stan::agrad::var> theta;
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

  ode_res = stan::math::integrate_ode(harm_osc, y0, t0,
                                      ts, theta, x, x_int, &msgs);

  EXPECT_NEAR(0.995029, ode_res[0][0].val(), 1e-5);
  EXPECT_NEAR(-0.0990884, ode_res[0][1].val(), 1e-5);

  EXPECT_NEAR(-0.421907, ode_res[99][0].val(), 1e-5);
  EXPECT_NEAR(0.246407, ode_res[99][1].val(), 1e-5);
  
  std::vector<double> grads;
  ode_res[99][1].grad(theta, grads);
}

TEST(integrate_ode, harm_osc_known_values_vd) {
  std::stringstream msgs;

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

  ode_res = stan::math::integrate_ode(harm_osc, y0, t0,
                                      ts, theta, x, x_int, &msgs);

  EXPECT_NEAR(0.995029, ode_res[0][0].val(), 1e-5);
  EXPECT_NEAR(-0.0990884, ode_res[0][1].val(), 1e-5);

  EXPECT_NEAR(-0.421907, ode_res[99][0].val(), 1e-5);
  EXPECT_NEAR(0.246407, ode_res[99][1].val(), 1e-5);
  
  std::vector<double> grads;
  ode_res[99][1].grad(y0, grads);
}

TEST(integrate_ode, harm_osc_known_values_vv) {
  std::stringstream msgs;

  harm_osc_ode_fun harm_osc;

  std::vector<stan::agrad::var> y0;
  std::vector<stan::agrad::var> theta;
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

  ode_res = stan::math::integrate_ode(harm_osc, y0, t0,
                                  ts, theta, x, x_int, &msgs);

  EXPECT_NEAR(0.995029, ode_res[0][0].val(), 1e-5);
  EXPECT_NEAR(-0.0990884, ode_res[0][1].val(), 1e-5);

  EXPECT_NEAR(-0.421907, ode_res[99][0].val(), 1e-5);
  EXPECT_NEAR(0.246407, ode_res[99][1].val(), 1e-5);
  
  std::vector<double> grads;
  std::vector<stan::agrad::var> variables;
  variables.insert(variables.end(), theta.begin(), theta.end());
  variables.insert(variables.end(), y0.begin(), y0.end());
  ode_res[99][1].grad(variables, grads);
}

template <typename T0, typename T1, typename T2>
inline
std::vector<typename stan::return_type<T1,T2>::type> 
lorenz_ode(const T0& t_in, // initial time
             const std::vector<T1>& y_in, //initial positions
             const std::vector<T2>& theta, // parameters
             const std::vector<double>& x, // double data
             const std::vector<int>& x_int) { // integer data
  std::vector<typename stan::return_type<T1,T2>::type> res;
  res.push_back(theta[0]*(y_in[1] - y_in[0]));
  res.push_back(theta[1]*y_in[0] - y_in[1] - y_in[0]*y_in[2]);
  res.push_back(-theta[2]*y_in[2] + y_in[0]*y_in[1]);
  return res;
}

struct lorenz_ode_fun {
  template <typename T0, typename T1, typename T2>
  inline 
  std::vector<typename stan::return_type<T1,T2>::type> 
  operator()(const T0& t_in, // initial time
             const std::vector<T1>& y_in, //initial positions
             const std::vector<T2>& theta, // parameters
             const std::vector<double>& x, // double data
             const std::vector<int>& x_int,
             std::ostream* msgs) const { // integer data
    return lorenz_ode(t_in, y_in, theta, x, x_int);
  }
};

TEST(integrate_ode, lorenz_finite_diff) {
  lorenz_ode_fun lorenz;

  std::vector<double> y0;
  std::vector<double> theta;
  double t0;
  std::vector<double> ts;

  t0 = 0;

  theta.push_back(10.0);
  theta.push_back(28.0);
  theta.push_back(8.0/3.0);
  y0.push_back(10.0);
  y0.push_back(1.0);
  y0.push_back(1.0);

  std::vector<double> x;
  std::vector<int> x_int;

  for (int i = 0; i < 100; i++)
    ts.push_back(0.1*(i+1));

  test_ode(lorenz, t0, ts, y0, theta, x, x_int, 1e-8, 1e-1);
}
