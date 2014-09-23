#include <gtest/gtest.h>
#include <test/unit/math/ode/harmonic_oscillator.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/math/ode/ode_system.hpp>

TEST(StanMathOde, ode_system_dv) {
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

TEST(StanMathOde, ode_system_vd) {
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
