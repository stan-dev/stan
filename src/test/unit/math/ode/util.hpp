#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/rev.hpp>

#include <stan/math/ode/solve_ode_diff_integrator.hpp>
#include <stan/math/ode/solve_ode.hpp>


//calculates finite diffs for solve_ode with varying parameters
template <typename F>
std::vector<std::vector<double> > finite_diff_params(const F& f,
                                                     const double& t_in,
                                                     const std::vector<double>& ts,
                                                     const std::vector<double>& y_in,
                                                     const std::vector<double>& theta,
                                                     const std::vector<double>& x,
                                                     const std::vector<int>& x_int,
                                                     const int& param_index,
                                                     const double& diff) {
  std::vector<double> theta_ub(theta.size());
  std::vector<double> theta_lb(theta.size());
  for (int i = 0; i < theta.size(); i++) {
    if (i == param_index) {
      theta_ub[i] = theta[i] + diff;
      theta_lb[i] = theta[i] - diff;
    } else {
      theta_ub[i] = theta[i];
      theta_lb[i] = theta[i];
    }
  }

  std::vector<std::vector<double> > ode_res_ub;
  std::vector<std::vector<double> > ode_res_lb;

  ode_res_ub = stan::math::solve_ode(f, y_in, t_in,
                                     ts, theta_ub, x, x_int);
  ode_res_lb = stan::math::solve_ode(f, y_in, t_in,
                                     ts, theta_lb, x, x_int);

  std::vector<std::vector<double> > results(ts.size());

  for (int i = 0; i < ode_res_ub.size(); i++) 
    for (int j = 0; j < ode_res_ub[j].size(); j++)
      results[i].push_back((ode_res_ub[i][j] - ode_res_lb[i][j]) / (2*diff));
  return results;
}
 

//calls solve_ode with initial positions as doubles and parameters as vars                
template <typename F>
inline std::vector<double> solve_ode_efficient_dv(const F& f,
                                                  const double& t_in,
                                                  const std::vector<double>& ts,
                                                  const std::vector<double>& y_in,
                                                  const std::vector<double>& theta,
                                                  const std::vector<double>& x,
                                                  const std::vector<int>& x_int,
                                                  const int& iteration_number,
                                                  const int& eqn_number) {
  std::vector<stan::agrad::var> theta_v;
  for (int i = 0; i < theta.size(); i++)
    theta_v.push_back(theta[i]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::solve_ode(f, y_in, t_in,
                                  ts, theta_v, x, x_int);
  
  std::vector<double> grads;
  ode_res[iteration_number][eqn_number].grad(theta_v, grads);
  return grads;
}                      

//test solve_ode with initial positions as doubles and parameters as vars 
//against finite differences
template <typename F>
void test_ode_dv(const F& f,
                 const double& t_in,
                 const std::vector<double>& ts,
                 const std::vector<double>& y_in,
                 const std::vector<double>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int,
                 const double& diff,
                 const double& diff2) {

  std::vector<std::vector<std::vector<double> > > finite_diff_res(theta.size());
  for (int i = 0; i < theta.size(); i++)
    finite_diff_res[i] = finite_diff_params(f, t_in, ts, y_in, theta, x, x_int, i, diff);

  std::vector<double> grads_eff;

  for (int i = 0; i < ts.size(); i++) {
    for (int j = 0; j < y_in.size(); j++) {
      grads_eff.clear();
      grads_eff = solve_ode_efficient_dv(f, t_in, ts, y_in, theta, x, x_int, i, j);

      for (int k = 0; k < theta.size(); k++)
        EXPECT_NEAR(grads_eff[k], finite_diff_res[k][i][j], diff2)
          << "Gradient of solve_ode failed with initial positions"
          << " known and parameters unknown at time index " << i
          << ", equation index " << j 
          << ", and parameter index: " << k;
    }
  }
}

//calculates finite diffs for solve_ode with varying initial positions
template <typename F>
std::vector<std::vector<double> > 
finite_diff_initial_position(const F& f,
                             const double& t_in,
                             const std::vector<double>& ts,
                             const std::vector<double>& y_in,
                             const std::vector<double>& theta,
                             const std::vector<double>& x,
                             const std::vector<int>& x_int,
                             const int& param_index,
                             const double& diff) {
  std::vector<double> y_in_ub(y_in.size());
  std::vector<double> y_in_lb(y_in.size());
  for (int i = 0; i < y_in.size(); i++) {
    if (i == param_index) {
      y_in_ub[i] = y_in[i] + diff;
      y_in_lb[i] = y_in[i] - diff;
    } else {
      y_in_ub[i] = y_in[i];
      y_in_lb[i] = y_in[i];
    }
  }

  std::vector<std::vector<double> > ode_res_ub;
  std::vector<std::vector<double> > ode_res_lb;

  ode_res_ub = stan::math::solve_ode(f, y_in_ub, t_in,
                                     ts, theta, x, x_int);
  ode_res_lb = stan::math::solve_ode(f, y_in_lb, t_in,
                                     ts, theta, x, x_int);

  std::vector<std::vector<double> > results(ts.size());

  for (int i = 0; i < ode_res_ub.size(); i++) 
    for (int j = 0; j < ode_res_ub[j].size(); j++)
      results[i].push_back((ode_res_ub[i][j] - ode_res_lb[i][j]) / (2*diff));
  return results;
}
 

//calls solve_ode with initial positions as vars and parameters as doubles                
template <typename F>
inline std::vector<double> solve_ode_efficient_vd(const F& f,
                                                  const double& t_in,
                                                  const std::vector<double>& ts,
                                                  const std::vector<double>& y_in,
                                                  const std::vector<double>& theta,
                                                  const std::vector<double>& x,
                                                  const std::vector<int>& x_int,
                                                  const int& iteration_number,
                                                  const int& eqn_number) {
  std::vector<stan::agrad::var> y_in_v;
  for (int i = 0; i < y_in.size(); i++)
    y_in_v.push_back(y_in[i]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::solve_ode(f, y_in_v, t_in,
                                  ts, theta, x, x_int);
  
  std::vector<double> grads;
  ode_res[iteration_number][eqn_number].grad(y_in_v, grads);
  return grads;
}                      


//test solve_ode with initial positions as vars and parameters as doubles 
//against finite differences
template <typename F>
void test_ode_vd(const F& f,
                 const double& t_in,
                 const std::vector<double>& ts,
                 const std::vector<double>& y_in,
                 const std::vector<double>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int,
                 const double& diff,
                 const double& diff2) {

  std::vector<std::vector<std::vector<double> > > finite_diff_res(y_in.size());
  for (int i = 0; i < y_in.size(); i++)
    finite_diff_res[i] = finite_diff_initial_position(f, t_in, ts, y_in, theta, x, x_int, i, diff);

  std::vector<double> grads_eff;

  for (int i = 0; i < ts.size(); i++) {
    for (int j = 0; j < y_in.size(); j++) {
      grads_eff.clear();
      grads_eff = solve_ode_efficient_vd(f, t_in, ts, y_in, theta, x, x_int, i, j);

      for (int k = 0; k < y_in.size(); k++)
        EXPECT_NEAR(grads_eff[k], finite_diff_res[k][i][j], diff2)
          << "Gradient of solve_ode failed with initial positions"
          << " unknown and parameters known at time index " << i
          << ", equation index " << j 
          << ", and parameter index: " << k;
    }
  }
}

//calls solve_ode with initial positions as vars and parameters as vars                
template <typename F>
inline std::vector<double> solve_ode_efficient_vv(const F& f,
                                                  const double& t_in,
                                                  const std::vector<double>& ts,
                                                  const std::vector<double>& y_in,
                                                  const std::vector<double>& theta,
                                                  const std::vector<double>& x,
                                                  const std::vector<int>& x_int,
                                                  const int& iteration_number,
                                                  const int& eqn_number) {
  std::vector<stan::agrad::var> y_in_v;
  for (int i = 0; i < y_in.size(); i++)
    y_in_v.push_back(y_in[i]);

  std::vector<stan::agrad::var> vars = y_in_v;

  std::vector<stan::agrad::var> theta_v;
  for (int i = 0; i < theta.size(); i++)
    theta_v.push_back(theta[i]);

  for (int i = 0; i < theta_v.size(); i++)
    vars.push_back(theta_v[i]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::solve_ode(f, y_in_v, t_in,
                                 ts, theta_v, x, x_int);
  
  std::vector<double> grads;
  ode_res[iteration_number][eqn_number].grad(vars, grads);
  return grads;
}           

//test solve_ode with initial positions as vars and parameters as vars 
//against finite differences
template <typename F>
void test_ode_vv(const F& f,
                 const double& t_in,
                 const std::vector<double>& ts,
                 const std::vector<double>& y_in,
                 const std::vector<double>& theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int,
                 const double& diff,
                 const double& diff2) {

  std::vector<std::vector<std::vector<double> > > finite_diff_res_y(y_in.size());
  for (int i = 0; i < y_in.size(); i++)
    finite_diff_res_y[i] = finite_diff_initial_position(f, t_in, ts, y_in,
                                                        theta, x, x_int, i, diff);

  std::vector<std::vector<std::vector<double> > > finite_diff_res_p(theta.size());
  for (int i = 0; i < theta.size(); i++)
    finite_diff_res_p[i] = finite_diff_params(f, t_in, ts, y_in, theta, x,
                                              x_int, i, diff);


  std::vector<double> grads_eff;

  for (int i = 0; i < ts.size(); i++) {
    for (int j = 0; j < y_in.size(); j++) {
      grads_eff.clear();
      grads_eff = solve_ode_efficient_vv(f, t_in, ts, y_in, theta, x, x_int, i, j);

      for (int k = 0; k < theta.size(); k++)
        EXPECT_NEAR(grads_eff[k+y_in.size()], finite_diff_res_p[k][i][j], diff2)
          << "Gradient of solve_ode failed with initial positions"
          << " unknown and parameters unknown for param at time index " << i
          << ", equation index " << j 
          << ", and parameter index: " << k;
      for (int k = 0; k < y_in.size(); k++)
        EXPECT_NEAR(grads_eff[k], finite_diff_res_y[k][i][j], diff2)
          << "Gradient of solve_ode failed with initial positions"
          << " unknown and parameters known for initial position at time index " << i
          << ", equation index " << j 
          << ", and parameter index: " << k;
    }
  }
}



// template <typename F>
// inline void solve_ode_diff_integrator(const F& f,
//                                       const std::vector<double>& y0_dbl,
//                                       const double& t0_dbl,
//                                       const std::vector<double>& ts_dbl,
//                                       const std::vector<double>& theta_dbl,
//                                       const std::vector<double>& x,
//                                       const std::vector<int>& x_int,
//                                       const int& iteration_number,
//                                       const int& eqn_number,
//                                       double& value,
//                                       std::vector<double>& gradients) {

//   std::vector<stan::agrad::var> y0;
//   for (int i = 0; i < y0_dbl.size(); i++)
//     y0.push_back(y0_dbl[i]);

//   double t0;
//   t0 = t0_dbl;

//   std::vector<double> ts;
//   for (int i = 0; i < ts_dbl.size(); i++)
//     ts.push_back(ts_dbl[i]);

//   std::vector<stan::agrad::var> theta;
//   for (int i = 0; i < theta_dbl.size(); i++)
//     theta.push_back(theta_dbl[i]);
  
//   std::vector<stan::agrad::var> vars;
//   for (int i = 0; i < theta.size(); i++)
//     vars.push_back(theta[i]);

//   for (int i = 0; i < y0.size(); i++)
//     vars.push_back(y0[i]);

//   std::vector<std::vector<stan::agrad::var> > ode_res;

//   ode_res = stan::math::solve_ode_diff_integrator(f, y0, t0,
//                                   ts, theta, x, x_int);
//   value = ode_res[iteration_number][eqn_number].val();
  
//   ode_res[iteration_number][eqn_number].grad(vars, gradients);
// }
