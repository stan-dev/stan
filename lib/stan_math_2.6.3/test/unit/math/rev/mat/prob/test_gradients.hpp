#include <cmath>
#include <vector>
#include <iomanip>

void test_grad_eq(Eigen::Matrix<double, -1, 1> grad_1,
                    Eigen::Matrix<double, -1, 1> grad_2) {
  for (int i = 0; i < grad_1.size(); ++i) 
    EXPECT_FLOAT_EQ(grad_1(i),grad_2(i));
}

double test_hess_eq(Eigen::Matrix<double, -1, -1> hess_1,
                    Eigen::Matrix<double, -1, -1> hess_2) {
  double accum(0.0);
  for (int i = 0; i < hess_1.size(); ++i)
    accum += fabs(hess_1(i) - hess_2(i));
  return accum;
}

double test_grad_hess_eq(std::vector<Eigen::Matrix<double, -1, -1> > g_hess_1,
                         std::vector<Eigen::Matrix<double, -1, -1> > g_hess_2) {
  double accum(0.0);
  for (size_t m = 0; m < g_hess_1.size(); ++m)
    for (int i = 0; i < g_hess_1[m].size(); ++i)
      accum += fabs(g_hess_1[m](i) - g_hess_2[m](i));
  return accum;
}


template <typename F>
std::vector<double> 
finite_diffs(const F& fun,
             const std::vector<double>& args,
             double epsilon = 1e-6) {
  std::vector<double> diffs(args.size());
  std::vector<double> args_plus = args;
  std::vector<double> args_minus = args;
  
  for (size_t i = 0; i < args.size(); ++i) {
    args_plus[i] += epsilon;
    args_minus[i] -= epsilon;
    diffs[i] = (fun(args_plus) - fun(args_minus)) / (2 * epsilon);
    args_plus[i] = args[i];
    args_minus[i] = args[i];
  }
  return diffs;
}

template <typename F>
std::vector<double>
grad(const F& fun,
     const std::vector<double>& args) {
  std::vector<stan::math::var> x;
  for (size_t i = 0; i < args.size(); ++i)
    x.push_back(args[i]);

  stan::math::var fx = fun(x);
  std::vector<double> grad;
  fx.grad(x,grad);
  return grad;
}


template <typename F>
void test_grad(const F& fun,
               const std::vector<double>& args) {
  using std::fabs;
  std::vector<double> diffs_finite = finite_diffs(fun,args);
  std::vector<double> diffs_var = grad(fun,args);
  EXPECT_EQ(diffs_finite.size(), diffs_var.size());
  for (size_t i = 0; i < args.size(); ++i) {
    double tolerance = 1e-6 * fmax(fabs(diffs_finite[i]), fabs(diffs_var[i])) + 1e-14;
    EXPECT_NEAR(diffs_finite[i], diffs_var[i], tolerance);
  }
}
