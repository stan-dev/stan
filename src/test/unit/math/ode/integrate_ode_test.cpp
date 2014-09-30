#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <vector>

#include <boost/numeric/odeint.hpp>

#include <stan/math/ode/coupled_ode_system.hpp>
#include <stan/math/ode/integrate_ode.hpp>

#include <test/unit/math/ode/util.hpp>
#include <test/unit/math/ode/harmonic_oscillator.hpp>
#include <test/unit/math/ode/lorenz.hpp>

TEST(StanMathOde_integrate_ode, harmonic_oscillator_known_values_dd) {
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
