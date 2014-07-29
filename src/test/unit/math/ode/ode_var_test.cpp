#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev.hpp>

#include <stan/math/ode/solve_ode.hpp>
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
                            const std::vector<int>& x_int) { // integer data
    return harm_osc_ode(t_in, y_in, theta, x, x_int);
  }
};


// namespace stan {
//   namespace agrad {
//     stan::agrad::var max(stan::agrad::var a, stan::agrad::var b) {
//       //return a > b ? a : b;
//       std::cout<<"whats wrong"<<std::endl;
//       return stan::agrad::fmax(a, b);
//     }

//     stan::agrad::var max(double a, stan::agrad::var b) {
//       //return a > b ? a : b;
//       std::cout<<"whats wrong2"<<std::endl;

//       return stan::agrad::fmax(a, b);
//     }

//     stan::agrad::var max(stan::agrad::var a, double b) {
//       //return a > b ? a : b;
//       std::cout<<"whats wrong3"<<std::endl;

//       return stan::agrad::fmax(a, b);
//     }
//    double max(double a, double b) {
//       //return a > b ? a : b;
//       std::cout<<"whats wrong3"<<std::endl;

//       return std::fmax(a, b);
//     }
//   }
// }

TEST(solve_ode, harm_osc) {
  harm_osc_ode_fun harm_osc;

  std::vector<stan::agrad::var> y0;
  std::vector<stan::agrad::var> theta;
  stan::agrad::var t0;
  std::vector<std::vector<stan::agrad::var> > ode_res;
  std::vector<stan::agrad::var> ts;

  stan::agrad::var gamma(0.15);

  theta.push_back(gamma);
  y0.push_back(1.0);
  y0.push_back(0.0);

  std::vector<double> x;
  std::vector<int> x_int;

  for (int i = 0; i < 100; i++)
    ts.push_back(0.1*(i+1));

  ode_res = stan::math::solve_ode(harm_osc, y0, t0,
                                  ts, theta, x, x_int);

  for (int i = 0; i < ode_res.size(); i++)
    std::cout<<"step "<<i<<": "<<ode_res[i][0]<<", "<<ode_res[i][1]<<std::endl;
}


