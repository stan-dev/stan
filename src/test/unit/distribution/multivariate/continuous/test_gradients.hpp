#include <cmath>
#include <vector>
#include <iomanip>
#include <stan/agrad/rev.hpp>

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
  std::vector<stan::agrad::var> x;
  for (size_t i = 0; i < args.size(); ++i)
    x.push_back(args[i]);

  stan::agrad::var fx = fun(x);
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
