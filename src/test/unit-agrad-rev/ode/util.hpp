#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include <stan/agrad/rev.hpp>

#include <stan/math/ode/integrate_ode.hpp>

//calculates finite diffs for integrate_ode with varying parameters
template <typename F>
std::vector<std::vector<double> > 
finite_diff_params(const F& f,
                   const double& t_in,
                   const std::vector<double>& ts,
                   const std::vector<double>& y_in,
                   const std::vector<double>& theta,
                   const std::vector<double>& x,
                   const std::vector<int>& x_int,
                   const int& param_index,
                   const double& diff) {
  std::stringstream msgs;
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

  ode_res_ub = stan::math::integrate_ode(f, y_in, t_in,
                                         ts, theta_ub, x, x_int, &msgs);
  ode_res_lb = stan::math::integrate_ode(f, y_in, t_in,
                                         ts, theta_lb, x, x_int, &msgs);

  std::vector<std::vector<double> > results(ts.size());

  for (int i = 0; i < ode_res_ub.size(); i++) 
    for (int j = 0; j < ode_res_ub[j].size(); j++)
      results[i].push_back((ode_res_ub[i][j] - ode_res_lb[i][j]) / (2*diff));
  return results;
}

//calculates finite diffs for integrate_ode with varying initial positions
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
  std::stringstream msgs;
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

  ode_res_ub = stan::math::integrate_ode(f, y_in_ub, t_in,
                                         ts, theta, x, x_int, &msgs);
  ode_res_lb = stan::math::integrate_ode(f, y_in_lb, t_in,
                                         ts, theta, x, x_int, &msgs);

  std::vector<std::vector<double> > results(ts.size());

  for (int i = 0; i < ode_res_ub.size(); i++) 
    for (int j = 0; j < ode_res_ub[j].size(); j++)
      results[i].push_back((ode_res_ub[i][j] - ode_res_lb[i][j]) / (2*diff));
  return results;
}

 
//test integrate_ode with initial positions as doubles and parameters as vars 
//against finite differences
template <typename F>
void test_ode_finite_diff_dv(const F& f,
                             const double& t_in,
                             const std::vector<double>& ts,
                             const std::vector<double>& y_in,
                             const std::vector<double>& theta,
                             const std::vector<double>& x,
                             const std::vector<int>& x_int,
                             const double& diff,
                             const double& diff2) {
  std::stringstream msgs;

  std::vector<std::vector<std::vector<double> > > finite_diff_res(theta.size());
  for (int i = 0; i < theta.size(); i++)
    finite_diff_res[i] = finite_diff_params(f, t_in, ts, y_in, theta, x, x_int, i, diff);

  std::vector<double> grads_eff;

  std::vector<stan::agrad::var> theta_v;
  for (int i = 0; i < theta.size(); i++)
    theta_v.push_back(theta[i]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::integrate_ode(f, y_in, t_in,
                                      ts, theta_v, x, x_int, &msgs);
  
  for (int i = 0; i < ts.size(); i++) {
    for (int j = 0; j < y_in.size(); j++) {
      grads_eff.clear();
      ode_res[i][j].grad(theta_v, grads_eff);

      for (int k = 0; k < theta.size(); k++)
        EXPECT_NEAR(grads_eff[k], finite_diff_res[k][i][j], diff2)
          << "Gradient of integrate_ode failed with initial positions"
          << " known and parameters unknown at time index " << i
          << ", equation index " << j 
          << ", and parameter index: " << k;

      stan::agrad::set_zero_all_adjoints();
    }
  }
}

//test integrate_ode with initial positions as vars and parameters as doubles 
//against finite differences
template <typename F>
void test_ode_finite_diff_vd(const F& f,
                             const double& t_in,
                             const std::vector<double>& ts,
                             const std::vector<double>& y_in,
                             const std::vector<double>& theta,
                             const std::vector<double>& x,
                             const std::vector<int>& x_int,
                             const double& diff,
                             const double& diff2) {
  std::stringstream msgs;

  std::vector<std::vector<std::vector<double> > > finite_diff_res(y_in.size());
  for (int i = 0; i < y_in.size(); i++)
    finite_diff_res[i] = finite_diff_initial_position(f, t_in, ts, y_in, theta, x, x_int, i, diff);

  std::vector<double> grads_eff;

  std::vector<stan::agrad::var> y_in_v;
  for (int k = 0; k < y_in.size(); k++)
    y_in_v.push_back(y_in[k]);

  std::vector<std::vector<stan::agrad::var> > ode_res;

  ode_res = stan::math::integrate_ode(f, y_in_v, t_in,
                                      ts, theta, x, x_int, &msgs);

  for (int i = 0; i < ts.size(); i++) {
    for (int j = 0; j < y_in.size(); j++) {
      grads_eff.clear();
      ode_res[i][j].grad(y_in_v, grads_eff);

      for (int k = 0; k < y_in.size(); k++)
        EXPECT_NEAR(grads_eff[k], finite_diff_res[k][i][j], diff2)
          << "Gradient of integrate_ode failed with initial positions"
          << " unknown and parameters known at time index " << i
          << ", equation index " << j 
          << ", and parameter index: " << k;

      stan::agrad::set_zero_all_adjoints();
    }
  }
}

//test integrate_ode with initial positions as vars and parameters as vars 
//against finite differences
template <typename F>
void test_ode_finite_diff_vv(const F& f,
                             const double& t_in,
                             const std::vector<double>& ts,
                             const std::vector<double>& y_in,
                             const std::vector<double>& theta,
                             const std::vector<double>& x,
                             const std::vector<int>& x_int,
                             const double& diff,
                             const double& diff2) {

  std::stringstream msgs;

  std::vector<std::vector<std::vector<double> > > finite_diff_res_y(y_in.size());
  for (int i = 0; i < y_in.size(); i++)
    finite_diff_res_y[i] = finite_diff_initial_position(f, t_in, ts, y_in,
                                                        theta, x, x_int, i, diff);

  std::vector<std::vector<std::vector<double> > > finite_diff_res_p(theta.size());
  for (int i = 0; i < theta.size(); i++)
    finite_diff_res_p[i] = finite_diff_params(f, t_in, ts, y_in, theta, x,
                                              x_int, i, diff);


  std::vector<double> grads_eff;
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

  ode_res = stan::math::integrate_ode(f, y_in_v, t_in,
                                      ts, theta_v, x, x_int, &msgs);

  for (int i = 0; i < ts.size(); i++) {
    for (int j = 0; j < y_in.size(); j++) {
      grads_eff.clear();
      ode_res[i][j].grad(vars, grads_eff);

      for (int k = 0; k < theta.size(); k++)
        EXPECT_NEAR(grads_eff[k+y_in.size()], finite_diff_res_p[k][i][j], diff2)
          << "Gradient of integrate_ode failed with initial positions"
          << " unknown and parameters unknown for param at time index " << i
          << ", equation index " << j 
          << ", and parameter index: " << k;
      for (int k = 0; k < y_in.size(); k++)
        EXPECT_NEAR(grads_eff[k], finite_diff_res_y[k][i][j], diff2)
          << "Gradient of integrate_ode failed with initial positions"
          << " unknown and parameters known for initial position at time index " << i
          << ", equation index " << j 
          << ", and parameter index: " << k;

      stan::agrad::set_zero_all_adjoints();
    }
  }
}

template <typename F, typename T1, typename T2>
void test_ode_exceptions(const F& f,
                         const double& t_in,
                         const std::vector<double>& ts,
                         const std::vector<T1>& y_in,
                         const std::vector<T2>& theta,
                         const std::vector<double>& x,
                         const std::vector<int>& x_int) {
  std::stringstream msgs;

  std::vector<T1> y_ = y_in;
  std::vector<T2> theta_ = theta;
  double t_ = t_in;
  std::vector<double> ts_ = ts;

  // y0.size() == 0 should throw
  y_.clear();
  EXPECT_THROW(stan::math::integrate_ode(f, y_, t_, ts_, theta_, x, x_int, &msgs),
               std::domain_error);
  y_ = y_in;

  // y0.size() =/= dy_dt.size() should throw
  y_.clear();
  for (int i = 0; i < y_in.size() - 1; i++)
    y_.push_back(y_in[i]);
  EXPECT_THROW(stan::math::integrate_ode(f, y_, t_, ts_, theta_, x, x_int, &msgs),
               std::domain_error);
  y_.clear();
  y_ = y_in;

  // ts.size() == 0 should throw  
  ts_.clear();
  EXPECT_THROW(stan::math::integrate_ode(f, y_, t_, ts_, theta_, x, x_int, &msgs),
               std::domain_error);

  // repeated values should throw
  ts_.clear();
  for (int i = 0; i < ts.size(); i++)
    ts_.push_back(t_in+1.0);
  EXPECT_THROW(stan::math::integrate_ode(f, y_, t_, ts_, theta_, x, x_int, &msgs),
               std::domain_error);

  // elements in ts need to be ordered
  ts_.clear();
  for (int i = 0; i < ts.size(); i++)
    ts_.push_back(ts[ts.size()-i]);
  EXPECT_THROW(stan::math::integrate_ode(f, y_, t_, ts_, theta_, x, x_int, &msgs),
               std::domain_error);

  // test t_in > ts (should throw)
  ts_.clear();
  ts_ = ts;
  t_ = ts[0] + 1.0;
  EXPECT_THROW(stan::math::integrate_ode(f, y_, t_, ts_, theta_, x, x_int, &msgs),
               std::domain_error);

  // test negative time values
  ts_.clear();
  for (int i = 1; i < 4; i++)
    ts_.push_back(-0.1*(6-i));
  t_ = ts_[0] - 1.0;
  EXPECT_NO_THROW(stan::math::integrate_ode(f, y_, t_, ts_, theta_, x, x_int, &msgs));
}

template <typename F>
void test_ode_exceptions_vd(const F& f,
                            const double& t_in,
                            const std::vector<double>& ts,
                            const std::vector<double>& y_in,
                            const std::vector<double>& theta,
                            const std::vector<double>& x,
                            const std::vector<int>& x_int) {
  std::vector<stan::agrad::var> y_var;
  for (int i = 0; i < y_in.size(); i++) 
    y_var.push_back(y_in[i]);
  test_ode_exceptions(f, t_in, ts, y_var, theta, x, x_int);
}
template <typename F>
void test_ode_exceptions_dv(const F& f,
                            const double& t_in,
                            const std::vector<double>& ts,
                            const std::vector<double>& y_in,
                            const std::vector<double>& theta,
                            const std::vector<double>& x,
                            const std::vector<int>& x_int) {
  std::vector<stan::agrad::var> theta_var;
  for (int i = 0; i < theta.size(); i++) 
    theta_var.push_back(theta[i]);
  test_ode_exceptions(f, t_in, ts, y_in, theta_var, x, x_int);
}
template <typename F>
void test_ode_exceptions_vv(const F& f,
                            const double& t_in,
                            const std::vector<double>& ts,
                            const std::vector<double>& y_in,
                            const std::vector<double>& theta,
                            const std::vector<double>& x,
                            const std::vector<int>& x_int) {
  std::vector<stan::agrad::var> y_var;
  for (int i = 0; i < y_in.size(); i++) 
    y_var.push_back(y_in[i]); 
  std::vector<stan::agrad::var> theta_var;
  for (int i = 0; i < theta.size(); i++) 
    theta_var.push_back(theta[i]);
  test_ode_exceptions(f, t_in, ts, y_var, theta_var, x, x_int);
}

template <typename F>
void test_ode(const F& f,
              const double& t_in,
              const std::vector<double>& ts,
              const std::vector<double>& y_in,
              const std::vector<double>& theta,
              const std::vector<double>& x,
              const std::vector<int>& x_int,
              const double& diff,
              const double& diff2) {
  test_ode_finite_diff_vd(f, t_in, ts, y_in, theta, x, x_int, diff, diff2);
  test_ode_finite_diff_dv(f, t_in, ts, y_in, theta, x, x_int, diff, diff2);
  test_ode_finite_diff_vv(f, t_in, ts, y_in, theta, x, x_int, diff, diff2);

  test_ode_exceptions_vd(f, t_in, ts, y_in, theta, x, x_int);
  test_ode_exceptions_dv(f, t_in, ts, y_in, theta, x, x_int);
  test_ode_exceptions_vv(f, t_in, ts, y_in, theta, x, x_int);
}
