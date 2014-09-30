#include <gtest/gtest.h>
#include <stan/agrad/rev.hpp>
//#include <stan/math/ode/coupled_ode_system.hpp>
#include <stan/agrad/rev/ode/coupled_ode_system.hpp>
#include <test/unit/math/ode/harmonic_oscillator.hpp>
#include <test/unit/math/ode/mock_ode_functor.hpp>

struct StanAgradRevOde : public ::testing::Test {
  std::stringstream msgs;
  std::vector<double> x;
  std::vector<int> x_int;
};


// ******************** DV ****************************
TEST_F(StanAgradRevOde, coupled_ode_system_dv) {
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
TEST_F(StanAgradRevOde, decouple_states_dv) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;

  harm_osc_ode_fun harm_osc;


  std::vector<double> y0(2);
  std::vector<var> theta(1);
  
  y0[0] = 1.0;
  y0[1] = 0.5;
  theta[0] = 0.15;
  
  coupled_ode_system<harm_osc_ode_fun, double, var> 
    coupled_system(harm_osc, y0, theta, x, x_int, &msgs);

  int T = 10;
  int k = 0;
  std::vector<std::vector<double> > ys_coupled(T);
  for (int t = 0; t < T; t++) {
    std::vector<double> coupled_state(coupled_system.size(), 0.0);
    for (int n = 0; n < coupled_system.size(); n++)
      coupled_state[n] = ++k;
    ys_coupled[t] = coupled_state;
  }

  std::vector<std::vector<var> > ys;
  ys = coupled_system.decouple_states(ys_coupled);

  ASSERT_EQ(T, ys.size());
  for (int t = 0; t < T; t++)
    ASSERT_EQ(2, ys[t].size());
  
  for (int t = 0; t < T; t++)
    for (int n = 0; n < 2; n++)
      EXPECT_FLOAT_EQ(ys_coupled[t][n], ys[t][n].val());

  // FIXME: add tests for derivatives
}
TEST_F(StanAgradRevOde, initial_state_dv) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<double> y0_d(N, 0.0);
  std::vector<var> theta_v(M, 0.0);

  for (int n = 0; n < N; n++)
    y0_d[n] = n+1;
  for (int m = 0; m < M; m++)
    theta_v[m] = 10 * (m+1);
     
  coupled_ode_system<mock_ode_functor, double, var>
    coupled_system_dv(base_ode, y0_d, theta_v, x, x_int, &msgs);

  std::vector<double> state = coupled_system_dv.initial_state();
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(y0_d[n], state[n])
      << "we don't need derivatives of y0; initial state gets the initial values";
  for (int n = N; n < state.size(); n++)
    EXPECT_FLOAT_EQ(0.0, state[n]);
}
TEST_F(StanAgradRevOde, size_dv) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<double> y0_d(N, 0.0);
  std::vector<var> theta_v(M, 0.0);

  coupled_ode_system<mock_ode_functor, double, var>
    coupled_system_dv(base_ode, y0_d, theta_v, x, x_int, &msgs);

  EXPECT_EQ(N + N*M, coupled_system_dv.size());
}










// ******************** VD ****************************

TEST_F(StanAgradRevOde, coupled_ode_system_vd) {
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
TEST_F(StanAgradRevOde, decouple_states_vd) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;

  harm_osc_ode_fun harm_osc;


  std::vector<var> y0(2);
  std::vector<double> theta(1);
  
  y0[0] = 1.0;
  y0[1] = 0.5;
  theta[0] = 0.15;
  
  coupled_ode_system<harm_osc_ode_fun, var, double>
    coupled_system(harm_osc, y0, theta, x, x_int, &msgs);

  int T = 10;
  int k = 0;
  std::vector<std::vector<double> > ys_coupled(T);
  for (int t = 0; t < T; t++) {
    std::vector<double> coupled_state(coupled_system.size(), 0.0);
    for (int n = 0; n < coupled_system.size(); n++)
      coupled_state[n] = ++k;
    ys_coupled[t] = coupled_state;
  }

  std::vector<std::vector<var> > ys;
  ys = coupled_system.decouple_states(ys_coupled);

  ASSERT_EQ(T, ys.size());
  for (int t = 0; t < T; t++)
    ASSERT_EQ(2, ys[t].size());
  
  for (int t = 0; t < T; t++)
    for (int n = 0; n < 2; n++)
      EXPECT_FLOAT_EQ(ys_coupled[t][n] + y0[n].val(), 
                      ys[t][n].val());

  // FIXME: add tests for derivatives
}
TEST_F(StanAgradRevOde, initial_state_vd) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<var> y0_v(N, 0.0);
  std::vector<double> theta_d(M, 0.0);

  for (int n = 0; n < N; n++)
    y0_v[n] = n+1;
  for (int m = 0; m < M; m++)
    theta_d[m] = 10 * (m+1);
     
  coupled_ode_system<mock_ode_functor, var, double>
    coupled_system_vd(base_ode, y0_v, theta_d, x, x_int, &msgs);

  std::vector<double> state;

  state = coupled_system_vd.initial_state();
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(0.0, state[n])
      << "we need derivatives of y0; initial state gets set to 0";
  for (int n = N; n < state.size(); n++)
    EXPECT_FLOAT_EQ(0.0, state[n]);
}
TEST_F(StanAgradRevOde, size_vd) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<var> y0_v(N, 0.0);
  std::vector<double> theta_d(M, 0.0);

  coupled_ode_system<mock_ode_functor, var, double>
    coupled_system_vd(base_ode, y0_v, theta_d, x, x_int, &msgs);

  EXPECT_EQ(N + N*N, coupled_system_vd.size());
}







// ******************** VV ****************************

TEST_F(StanAgradRevOde, coupled_ode_system_vv) {
  // FIXME: no tests in here at all?
}
TEST_F(StanAgradRevOde, decouple_states_vv) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;

  harm_osc_ode_fun harm_osc;

  std::vector<var> y0(2);
  std::vector<var> theta(1);
  
  y0[0] = 1.0;
  y0[1] = 0.5;
  theta[0] = 0.15;
  
  coupled_ode_system<harm_osc_ode_fun, var, var>
    coupled_system(harm_osc, y0, theta, x, x_int, &msgs);

  int T = 10;
  int k = 0;
  std::vector<std::vector<double> > ys_coupled(T);
  for (int t = 0; t < T; t++) {
    std::vector<double> coupled_state(coupled_system.size(), 0.0);
    for (int n = 0; n < coupled_system.size(); n++)
      coupled_state[n] = ++k;
    ys_coupled[t] = coupled_state;
  }

  std::vector<std::vector<var> > ys;
  ys = coupled_system.decouple_states(ys_coupled);

  ASSERT_EQ(T, ys.size());
  for (int t = 0; t < T; t++)
    ASSERT_EQ(2, ys[t].size());
  
  for (int t = 0; t < T; t++)
    for (int n = 0; n < 2; n++)
      EXPECT_FLOAT_EQ(ys_coupled[t][n] + y0[n].val(), 
                      ys[t][n].val());

  // FIXME: add tests for derivatives
}
TEST_F(StanAgradRevOde, initial_state_vv) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<var> y0_v(N, 0.0);
  std::vector<var> theta_v(M, 0.0);

  for (int n = 0; n < N; n++)
    y0_v[n] = n+1;
  for (int m = 0; m < M; m++)
    theta_v[m] = 10 * (m+1);
     
  coupled_ode_system<mock_ode_functor, var, var>
    coupled_system_vv(base_ode, y0_v, theta_v, x, x_int, &msgs);

  std::vector<double>  state = coupled_system_vv.initial_state();
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(0.0, state[n])
      << "we need derivatives of y0; initial state gets set to 0";
  for (int n = N; n < state.size(); n++)
    EXPECT_FLOAT_EQ(0.0, state[n]);
}
TEST_F(StanAgradRevOde, size_vv) {
  using stan::math::coupled_ode_system;
  using stan::agrad::var;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<var> y0_v(N, 0.0);
  std::vector<var> theta_v(M, 0.0);

  coupled_ode_system<mock_ode_functor, var, var>
    coupled_system_vv(base_ode, y0_v, theta_v, x, x_int, &msgs);

  EXPECT_EQ(N + N*N + N*M, coupled_system_vv.size());
}


















