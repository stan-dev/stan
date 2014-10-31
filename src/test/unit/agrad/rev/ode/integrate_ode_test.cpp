#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <vector>

#include <boost/numeric/odeint.hpp>

#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/ode/coupled_ode_system.hpp>

#include <stan/math/ode/coupled_ode_system.hpp>
#include <stan/math/ode/integrate_ode.hpp>
#include <stan/math/functions/promote_scalar.hpp>

#include <test/unit/agrad/rev/ode/util.hpp>

#include <test/unit/math/ode/harmonic_oscillator.hpp>
#include <test/unit/math/ode/lorenz.hpp>


template <typename F, typename T_y0, typename T_theta>
void sho_value_test(F harm_osc,
                    std::vector<double>& y0,
                    double t0,
                    std::vector<double>& ts,
                    std::vector<double>& theta,
                    std::vector<double>& x,
                    std::vector<int>& x_int) {
  
  using stan::agrad::var;
  using stan::math::promote_scalar;

  std::vector<std::vector<var> >  ode_res_vd
    = stan::math::integrate_ode(harm_osc, promote_scalar<T_y0>(y0), t0,
                                ts, promote_scalar<T_theta>(theta), x, x_int,
                                0);
  EXPECT_NEAR(0.995029, ode_res_vd[0][0].val(), 1e-5);
  EXPECT_NEAR(-0.0990884, ode_res_vd[0][1].val(), 1e-5);

  EXPECT_NEAR(-0.421907, ode_res_vd[99][0].val(), 1e-5);
  EXPECT_NEAR(0.246407, ode_res_vd[99][1].val(), 1e-5);

}

void sho_finite_diff_test(double t0) {
  using stan::agrad::var;  
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

  test_ode(harm_osc, t0, ts, y0, theta, x, x_int, 1e-8,1e-4);

  sho_value_test<harm_osc_ode_fun,double,var>(harm_osc, y0, t0, ts, theta, x, x_int);
  sho_value_test<harm_osc_ode_fun,var,double>(harm_osc, y0, t0, ts, theta, x, x_int);
  sho_value_test<harm_osc_ode_fun,var,var>(harm_osc, y0, t0, ts, theta, x, x_int);
}

void sho_data_finite_diff_test(double t0) {
  using stan::agrad::var;  
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

  test_ode(harm_osc, t0, ts, y0, theta, x, x_int, 1e-8,1e-4);

  sho_value_test<harm_osc_ode_data_fun,double,var>(harm_osc, y0, t0, ts, theta, x, x_int);
  sho_value_test<harm_osc_ode_data_fun,var,double>(harm_osc, y0, t0, ts, theta, x, x_int);
  sho_value_test<harm_osc_ode_data_fun,var,var>(harm_osc, y0, t0, ts, theta, x, x_int);
  
}


TEST(StanAgradRevOde_integrate_ode, harmonic_oscillator_finite_diff) {
  sho_finite_diff_test(0);
  sho_finite_diff_test(1.0);
  sho_finite_diff_test(-1.0);

  sho_data_finite_diff_test(0);
  sho_data_finite_diff_test(1.0);
  sho_data_finite_diff_test(-1.0);
}



TEST(StanAgradRevOde_integrate_ode, lorenz_finite_diff) {
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
