#include <gtest/gtest.h>
#include <stan/math/prim/arr/functor/coupled_ode_system.hpp>
#include <test/unit/util.hpp>
#include <test/unit/math/prim/arr/functor/harmonic_oscillator.hpp>
#include <test/unit/math/prim/arr/functor/mock_ode_functor.hpp>
#include <test/unit/math/prim/arr/functor/mock_throwing_ode_functor.hpp>

struct StanMathOde : public ::testing::Test {
  std::stringstream msgs;
  std::vector<double> x;
  std::vector<int> x_int;
};

TEST_F(StanMathOde, decouple_states_dd) {
  using stan::math::coupled_ode_system;

  harm_osc_ode_fun harm_osc;

  std::vector<double> y0(2);
  std::vector<double> theta(1);
  
  y0[0] = 1.0;
  y0[1] = 0.5;
  theta[0] = 0.15;
  
  coupled_ode_system<harm_osc_ode_fun, double, double> 
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

  std::vector<std::vector<double> > ys;
  ys = coupled_system.decouple_states(ys_coupled);
  
  ASSERT_EQ(T, ys.size());
  for (int t = 0; t < T; t++)
    ASSERT_EQ(2, ys[t].size());

  for (int t = 0; t < T; t++)
    for (int n = 0; n < 2; n++)
      EXPECT_FLOAT_EQ(ys_coupled[t][n], ys[t][n])
        << "(" << n << "," << t << "): "
        << "for (double, double) the coupled system is the base system";
}

TEST_F(StanMathOde, initial_state_dd) {
  using stan::math::coupled_ode_system;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<double> y0_d(N, 0.0);
  std::vector<double> theta_d(M, 0.0);

  for (int n = 0; n < N; n++)
    y0_d[n] = n+1;
  for (int m = 0; m < M; m++)
    theta_d[m] = 10 * (m+1);
     
  coupled_ode_system<mock_ode_functor, double, double>
    coupled_system_dd(base_ode, y0_d, theta_d, x, x_int, &msgs);

  std::vector<double> state  = coupled_system_dd.initial_state();
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(y0_d[n], state[n])
      << "we don't need derivatives of y0; initial state gets the initial values";
  for (size_t n = N; n < state.size(); n++)
    EXPECT_FLOAT_EQ(0.0, state[n]);
}

TEST_F(StanMathOde, size) {
  using stan::math::coupled_ode_system;
  mock_ode_functor base_ode;

  const int N = 3;
  const int M = 4;

  std::vector<double> y0_d(N, 0.0);
  std::vector<double> theta_d(M, 0.0);

  coupled_ode_system<mock_ode_functor, double, double>
    coupled_system_dd(base_ode, y0_d, theta_d, x, x_int, &msgs);

  EXPECT_EQ(N, coupled_system_dd.size());
}


TEST_F(StanMathOde, recover_exception) {
  using stan::math::coupled_ode_system;
  std::string message = "ode throws";

  const int N = 3;
  const int M = 4;
  
  mock_throwing_ode_functor<std::logic_error> throwing_ode(message);
  
  std::vector<double> y0_d(N, 0.0);
  std::vector<double> theta_v(M, 0.0);
  
  coupled_ode_system<mock_throwing_ode_functor<std::logic_error>, double, double>
    coupled_system_dv(throwing_ode, y0_d, theta_v, x, x_int, &msgs);
    
  std::vector<double> y(3,0);
  std::vector<double> dy_dt(3,0);
  
  double t = 10;
  
  EXPECT_THROW_MSG(coupled_system_dv(y, dy_dt, t),
                   std::logic_error,
                   message);
}
