#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev.hpp>

#include <stan/math/ode/solve_ode2.hpp>
#include <stan/math.hpp>

typedef std::vector< stan::agrad::var > state_type;

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

// TEST(solve_ode2, harm_osc) {
//   harm_osc_ode_fun harm_osc;

//   std::vector<stan::agrad::var> y0;
//   std::vector<stan::agrad::var> theta;
//   stan::agrad::var t0;
//   std::vector<std::vector<stan::agrad::var> > ode_res;
//   std::vector<stan::agrad::var> ts;

//   stan::agrad::var gamma(0.15);
//   t0 = 0;

//   theta.push_back(gamma);
//   y0.push_back(1.0);
//   y0.push_back(0.0);

//   std::vector<double> x;
//   std::vector<int> x_int;

//   for (int i = 0; i < 100; i++)
//     ts.push_back(0.1*(i+1));

//   ode_res = stan::math::solve_ode(harm_osc, y0, t0,
//                                   ts, theta, x, x_int);

//   EXPECT_NEAR(0.995029, ode_res[1][0].val(), 1e-6);
//   EXPECT_NEAR(-0.0990884, ode_res[1][1].val(), 1e-6);

//   EXPECT_NEAR(-0.421907, ode_res[100][0].val(), 1e-6);
//   EXPECT_NEAR(0.246407, ode_res[100][1].val(), 1e-6);

//   std::vector<double> grad;
//   ode_res[100][0].grad(theta,grad);
//   std::cout << grad[0] << std::endl;
// }


