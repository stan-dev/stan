#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fwd/functions/log_sum_exp.hpp>


template <int R, int C>
void log_sum_exp_test(const Eigen::Matrix<double,R,C>& x) {
  using std::exp;
  using stan::agrad::fvar;
  using stan::math::log_sum_exp;
  for (int n = 0; n < x.size(); ++n) {
    // for d/d.x[n]
    std::vector<fvar<double> > xv(x.size());
    for (int i = 0; i < x.size(); ++i)
      xv[i] = x(i);
    xv[n].d_ = 2.3;
    fvar<double> sum_exp = 0;
    for (int i = 0; i < x.size(); ++i)
      sum_exp += exp(xv[i]);
    fvar<double> log_sum_exp_expected = log(sum_exp);
    double val_expected = log_sum_exp_expected.val_;
    double deriv_expected = log_sum_exp_expected.d_;
  
    Eigen::Matrix<fvar<double>,R,C> xv2(x.rows(),x.cols());
    for (int i = 0; i < x.size(); ++i)
      xv2(i) = x(i);
    xv2(n).d_ = 2.3;
    fvar<double> log_sum_exp_fvar = log_sum_exp(xv2);
    double val = log_sum_exp_fvar.val_;
    double deriv = log_sum_exp_fvar.d_;
    
    EXPECT_FLOAT_EQ(val_expected, val);
    EXPECT_FLOAT_EQ(deriv_expected, deriv);
  }
}

TEST(AgradRevLogSumExp,matrix) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,1> a(1);
  a << 1.3;
  log_sum_exp_test(a);

  Matrix<double,Dynamic,1> b(4);
  b << 1, 2, 3, 4;
  log_sum_exp_test(b);

  Matrix<double,1,Dynamic> c(3);
  c << -1, -4, -2;
  log_sum_exp_test(c);
  
  Matrix<double,Dynamic,Dynamic> d(2,3);
  d << 1, 2, 3, 5, 9, -3;
  log_sum_exp_test(d);
}
