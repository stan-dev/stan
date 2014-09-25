#include <gtest/gtest.h>
#include <test/unit/math/ode/harmonic_oscillator.hpp>
#include <test/unit/math/ode/mock_ode_functor.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/math/ode/coupled_ode_system.hpp>

struct StanMathOde : public ::testing::Test {
  std::stringstream msgs;
  std::vector<double> x;
  std::vector<int> x_int;
};

TEST_F(StanMathOde, coupled_ode_system_dv) {
  using stan::math::coupled_ode_system;

  harm_osc_ode_fun harm_osc;

  std::vector<stan::agrad::var> theta;
  std::vector<double> coupled_y0;
  std::vector<double> y0;
  double t0;
  std::vector<double> dy_dt;

  double gamma(0.15);
  t0 = 0;

  theta.push_back(gamma);
  y0.push_back(1.0);
  y0.push_back(0.5);
  
  coupled_y0.push_back(1.0);
  coupled_y0.push_back(0.5);
  coupled_y0.push_back(1.0);
  coupled_y0.push_back(2.0);

  coupled_ode_system<harm_osc_ode_fun, double, stan::agrad::var> 
    system(harm_osc, y0, theta, x, x_int, &msgs);

  system(coupled_y0, dy_dt, t0);

  EXPECT_FLOAT_EQ(0.5, dy_dt[0]);
  EXPECT_FLOAT_EQ(-1.075, dy_dt[1]);
  EXPECT_FLOAT_EQ(2, dy_dt[2]);
  EXPECT_FLOAT_EQ(-1.8, dy_dt[3]);
}

TEST_F(StanMathOde, coupled_ode_system_vd) {
  using stan::math::coupled_ode_system;

  harm_osc_ode_fun harm_osc;

  std::vector<double> theta;
  std::vector<double> coupled_y0;
  std::vector<stan::agrad::var> y0_var;
  std::vector<double> y0_adj;
  double t0;
  std::vector<double> dy_dt;

  double gamma(0.15);
  t0 = 0;

  theta.push_back(gamma);
  
  coupled_y0.push_back(1.0);
  coupled_y0.push_back(0.5);
  coupled_y0.push_back(1.0);
  coupled_y0.push_back(3.0);
  coupled_y0.push_back(2.0);
  coupled_y0.push_back(5.0);

  y0_var.push_back(1.0);
  y0_var.push_back(0.5);
  
  coupled_ode_system<harm_osc_ode_fun, stan::agrad::var, double> 
    system(harm_osc, y0_var, theta, x, x_int, &msgs);

  system(coupled_y0, dy_dt, t0);

  EXPECT_FLOAT_EQ(1.0, dy_dt[0]);
  EXPECT_FLOAT_EQ(-2.0 - 0.15*1.0, dy_dt[1]);
  EXPECT_FLOAT_EQ(0+1.0*0+3.0*1+0, dy_dt[2]);
  EXPECT_FLOAT_EQ(-1.0-1.0*1.0-0.15*3.0, dy_dt[3]);
  EXPECT_FLOAT_EQ(1.0+2.0*0+5.0*1.0, dy_dt[4]);
  EXPECT_FLOAT_EQ(-0.15-1.0*2.0-0.15*5.0, dy_dt[5]);
}


TEST_F(StanMathOde, coupled_size) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<double> y0_d(N, 0.0);
  std::vector<var> y0_v(N, 0.0);
  std::vector<double> theta_d(M, 0.0);
  std::vector<var> theta_v(M, 0.0);

  coupled_ode_system<mock_ode_functor, double, double>
    coupled_system_dd(base_ode, y0_d, theta_d, x, x_int, &msgs);
  coupled_ode_system<mock_ode_functor, double, var>
    coupled_system_dv(base_ode, y0_d, theta_v, x, x_int, &msgs);
  coupled_ode_system<mock_ode_functor, var, double>
    coupled_system_vd(base_ode, y0_v, theta_d, x, x_int, &msgs);
  coupled_ode_system<mock_ode_functor, var, var>
    coupled_system_vv(base_ode, y0_v, theta_v, x, x_int, &msgs);


  EXPECT_EQ(N, coupled_system_dd.size());
  EXPECT_EQ(N + N*M, coupled_system_dv.size());
  EXPECT_EQ(N + N*N, coupled_system_vd.size());
  EXPECT_EQ(N + N*N + N*M, coupled_system_vv.size());
}
