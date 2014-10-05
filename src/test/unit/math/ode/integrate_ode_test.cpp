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

#include <test/unit/util.hpp>


template <typename F>
void sho_death_test(F harm_osc,
                        std::vector<double>& y0,
                        double t0,
                        std::vector<double>& ts,
                        std::vector<double>& theta,
                        std::vector<double>& x,
                        std::vector<int>& x_int) {
  EXPECT_DEATH(stan::math::integrate_ode(harm_osc, y0, t0,
                                         ts, theta, x, x_int,0),
               "");
}


template <typename E, typename F>
void sho_exception_test(F harm_osc,
                        std::vector<double>& y0,
                        double t0,
                        std::vector<double>& ts,
                        std::vector<double>& theta,
                        std::vector<double>& x,
                        std::vector<int>& x_int,
                        std::string expected_message) {
  try {
    stan::math::integrate_ode(harm_osc, y0, t0,
                              ts, theta, x, x_int,0);
  } catch (const E& e) {
    EXPECT_EQ(1, count_matches(expected_message, e.what()))
      << "expected message: " << expected_message << std::endl
      << "found message:    " << e.what();
    return;
  }
  FAIL() << "didn't throw an exception of the correct type";
}


template <typename F>
void sho_value_test(F harm_osc,
                    std::vector<double>& y0,
                    double t0,
                    std::vector<double>& ts,
                    std::vector<double>& theta,
                    std::vector<double>& x,
                    std::vector<int>& x_int) {
  
  std::vector<std::vector<double> >  ode_res_vd
    = stan::math::integrate_ode(harm_osc, y0, t0,
                                ts, theta, x, x_int,
                                0);
  EXPECT_NEAR(0.995029, ode_res_vd[0][0], 1e-5);
  EXPECT_NEAR(-0.0990884, ode_res_vd[0][1], 1e-5);

  EXPECT_NEAR(-0.421907, ode_res_vd[99][0], 1e-5);
  EXPECT_NEAR(0.246407, ode_res_vd[99][1], 1e-5);
}

void sho_test(double t0) {
  harm_osc_ode_fun harm_osc;

  std::vector<double> theta;
  theta.push_back(0.15);

  std::vector<double> y0;
  y0.push_back(1.0);
  y0.push_back(0.0);

  std::vector<double> ts;
  for (int i = 0; i < 100; i++)
    ts.push_back(t0 + 0.1 * (i + 1));

  std::vector<double> x;
  std::vector<int> x_int;

  sho_value_test(harm_osc, y0, t0, ts, theta, x, x_int);
}


void sho_data_test(double t0) {
  harm_osc_ode_data_fun harm_osc;

  std::vector<double> theta;
  theta.push_back(0.15);

  std::vector<double> y0;
  y0.push_back(1.0);
  y0.push_back(0.0);

  std::vector<double> ts;
  for (int i = 0; i < 100; i++)
    ts.push_back(t0 + 0.1 * (i + 1));

  std::vector<double> x(3,1);
  std::vector<int> x_int(2,0);

  sho_value_test(harm_osc, y0, t0, ts, theta, x, x_int);
}


TEST(StanMathOde_integrate_ode, harmonic_oscillator) {
  sho_test(0.0);
  sho_test(1.0);
  sho_test(-1.0);

  sho_data_test(0.0);
  sho_data_test(1.0);
  sho_data_test(-1.0);
}

TEST(StanMathOde_integrate_ode, error_conditions) {
  harm_osc_ode_data_fun harm_osc;

  std::vector<double> theta;
  theta.push_back(0.15);

  std::vector<double> y0;
  y0.push_back(1.0);
  y0.push_back(0.0);

  double t0 = 0;

  std::vector<double> ts;
  for (int i = 0; i < 100; i++)
    ts.push_back(t0 + 0.1 * (i + 1));

  std::vector<double> x(3,1);
  std::vector<int> x_int(2,0);


  std::vector<double> y0_bad;
  sho_exception_test<std::domain_error>(harm_osc, y0_bad, t0, ts, theta, x, x_int, "has size 0");
  sho_exception_test<std::domain_error>(harm_osc, y0_bad, t0, ts, theta, x, x_int, "initial state");
  
  
  double t0_bad = 2.0;
  sho_exception_test<std::domain_error>(harm_osc, y0, t0_bad, ts, theta, x, x_int, "initial time is 2, but must be less than 0.1");

  std::vector<double> ts_bad;
  sho_exception_test<std::domain_error>(harm_osc, y0, t0, ts_bad, theta, x, x_int, "has size 0");
  sho_exception_test<std::domain_error>(harm_osc, y0, t0, ts_bad, theta, x, x_int, "times");

  ts_bad.push_back(3);
  ts_bad.push_back(1);
  sho_exception_test<std::domain_error>(harm_osc, y0, t0, ts_bad, theta, x, x_int, "times is not a valid ordered vector");


  std::vector<double> theta_bad;
  sho_exception_test<std::out_of_range>(harm_osc, y0, t0, ts, theta_bad, x, x_int, "vector");
  
  std::vector<double> x_bad;
  sho_exception_test<std::out_of_range>(harm_osc, y0, t0, ts, theta, x_bad, x_int, "vector");

  std::vector<int> x_int_bad;
  sho_exception_test<std::out_of_range>(harm_osc, y0, t0, ts, theta, x, x_int_bad, "vector");
  
}
