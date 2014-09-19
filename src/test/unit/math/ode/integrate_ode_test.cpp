#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev.hpp>

#include <stan/math/ode/integrate_ode.hpp>

#include <test/unit/math/ode/util.hpp>

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
             const std::vector<int>& x_int,
             std::ostream* msgs) const { // integer data
    return harm_osc_ode(t_in, y_in, theta, x, x_int);
  }
};

TEST(integrate_ode, ode_system_dv) {
  std::stringstream msgs;
  
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

  ode_system<harm_osc_ode_fun, double, stan::agrad::var> 
    system(harm_osc, y0,theta, x, x_int,2,&msgs);

  system(y0, dy_dt, t0);

  EXPECT_FLOAT_EQ(0.5, dy_dt[0]);
  EXPECT_FLOAT_EQ(-1.075, dy_dt[1]);
  EXPECT_FLOAT_EQ(2, dy_dt[2]);
  EXPECT_FLOAT_EQ(-1.8, dy_dt[3]);
}

TEST(integrate_ode, ode_system_vd) {
  std::stringstream msgs;

  using stan::math::ode_system;

  harm_osc_ode_fun harm_osc;

  std::vector<double> theta;
  std::vector<double> y0;
  std::vector<double> y0_adj;
  double t0;
  std::vector<double> dy_dt;

  double gamma(0.15);
  t0 = 0;

  theta.push_back(gamma);
  y0.push_back(1.0);
  y0.push_back(0.5);
  y0.push_back(1.0);
  y0.push_back(3.0);
  y0.push_back(2.0);
  y0.push_back(5.0);

  std::vector<double> x;
  std::vector<int> x_int;

  ode_system<harm_osc_ode_fun, stan::agrad::var, double> 
    system(harm_osc, y0,theta, x, x_int,2,&msgs);

  system(y0, dy_dt, t0);

  EXPECT_FLOAT_EQ(1.0, dy_dt[0]);
  EXPECT_FLOAT_EQ(-2.0 - 0.15*1.0, dy_dt[1]);
  EXPECT_FLOAT_EQ(0+1.0*0+3.0*1+0, dy_dt[2]);
  EXPECT_FLOAT_EQ(-1.0-1.0*1.0-0.15*3.0, dy_dt[3]);
  EXPECT_FLOAT_EQ(1.0+2.0*0+5.0*1.0, dy_dt[4]);
  EXPECT_FLOAT_EQ(-0.15-1.0*2.0-0.15*5.0, dy_dt[5]);
}

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

TEST(integrate_ode, harm_osc_known_values) {
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
  ode_res[99][1].grad(theta, grads);
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
