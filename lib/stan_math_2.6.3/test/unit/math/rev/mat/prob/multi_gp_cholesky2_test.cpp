#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <gtest/gtest.h>

#include <stan/math/prim/mat/prob/multi_gp_cholesky_log.hpp>
#include <stan/math/rev/mat/fun/to_var.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <stan/math/rev/mat/fun/dot_self.hpp>
#include <stan/math/rev/mat/fun/columns_dot_self.hpp>

// UTILITY FUNCTIONS FOR TESTING
#include <vector>
#include <test/unit/math/rev/mat/prob/expect_eq_diffs.hpp>
#include <test/unit/math/rev/mat/prob/test_gradients.hpp>
#include <test/unit/math/prim/mat/prob/agrad_distributions_multi_gp_cholesky.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

template <typename T_y, typename T_scale, typename T_w> 
void expect_propto(T_y y1, T_scale L1, T_w w1, 
                   T_y y2, T_scale L2, T_w w2, 
                   std::string message = "") {
  expect_eq_diffs(stan::math::multi_gp_cholesky_log<false>(y1,L1,w1),
                  stan::math::multi_gp_cholesky_log<false>(y2,L2,w2),
                  stan::math::multi_gp_cholesky_log<true>(y1,L1,w1),
                  stan::math::multi_gp_cholesky_log<true>(y2,L2,w2),
                  message);
}

using stan::math::var;
using stan::math::to_var;


TEST_F(agrad_distributions_multi_gp_cholesky,Propto) {
  expect_propto(to_var(y),to_var(L),to_var(w),
                to_var(y2),to_var(L2),to_var(w2),
                "All vars: y, w, sigma");
}
TEST_F(agrad_distributions_multi_gp_cholesky,ProptoY) {
  expect_propto(to_var(y),L,w,
                to_var(y2),L,w,
                "var: y");

}
TEST_F(agrad_distributions_multi_gp_cholesky,ProptoYMu) {
  expect_propto(to_var(y),L,to_var(w),
                to_var(y2),L,to_var(w2),
                "var: y and w");
}
TEST_F(agrad_distributions_multi_gp_cholesky,ProptoYSigma) {
  expect_propto(to_var(y),to_var(L),w,
                to_var(y2),to_var(L2),w,
                "var: y and sigma");
}
TEST_F(agrad_distributions_multi_gp_cholesky,ProptoMu) {
  expect_propto(y,L,to_var(w),
                y,L,to_var(w2),
                "var: w");
}
TEST_F(agrad_distributions_multi_gp_cholesky,ProptoMuSigma) {
  expect_propto(y,to_var(L),to_var(w),
                y,to_var(L2),to_var(w2),
                "var: w and sigma");
}
TEST_F(agrad_distributions_multi_gp_cholesky,ProptoSigma) {
  expect_propto(y,to_var(L),w,
                y,to_var(L2),w,
                "var: sigma");
}


TEST(ProbDistributionsMultiGPCholesky,MultiGPCholeskyVar) {
  using stan::math::var;
  Matrix<var,Dynamic,Dynamic> y(3,3);
  y <<  2.0, -2.0, 11.0,
       -4.0, 0.0, 2.0,
        1.0, 5.0, 3.3;
  Matrix<var,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 3.0;
  Matrix<var,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
          -3.0,  4.0, 0.0,
           0.0, 0.0, 5.0;
  Matrix<var,Dynamic,Dynamic> L = Sigma.llt().matrixL();
  EXPECT_FLOAT_EQ(-46.087162, stan::math::multi_gp_cholesky_log(y,L,w).val());
}

TEST(ProbDistributionsMultiGPCholesky,MultiGPCholeskyGradientUnivariate) {
  using stan::math::var;
  using std::vector;
  using Eigen::VectorXd;
  using stan::math::multi_gp_cholesky_log;
  
  Matrix<var,Dynamic,Dynamic> y_var(1,1);
  y_var << 2.0;

  Matrix<var,Dynamic,1> w_var(1,1);
  w_var << 1.0;

  Matrix<var,Dynamic,Dynamic> L_var(1,1);
  L_var(0,0) = 3.0;

  std::vector<var> x;
  x.push_back(y_var(0));
  x.push_back(w_var(0));
  x.push_back(L_var(0,0));

  var lp = stan::math::multi_gp_cholesky_log(y_var,L_var,w_var);
  vector<double> grad;
  lp.grad(x,grad);

  // ===================================


  Matrix<double,Dynamic,Dynamic> y(1,1);
  y << 2.0;

  Matrix<double,Dynamic,1> w(1,1);
  w << 1.0;

  Matrix<double,Dynamic,Dynamic> L(1,1);
  L << 3.0;

  double epsilon = 1e-6;


  Matrix<double,Dynamic,Dynamic> y_m(1,1);
  Matrix<double,Dynamic,Dynamic> y_p(1,1);
  y_p(0) = y(0) + epsilon;
  y_m(0) = y(0) - epsilon;
  double grad_diff 
    =  (multi_gp_cholesky_log(y_p,L,w) - multi_gp_cholesky_log(y_m,L,w)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[0]);

  Matrix<double,Dynamic,1> w_m(1,1);
  Matrix<double,Dynamic,1> w_p(1,1);
  w_p[0] = w[0] + epsilon;
  w_m[0] = w[0] - epsilon;
  grad_diff 
    =  (multi_gp_cholesky_log(y,L,w_p) - multi_gp_cholesky_log(y,L,w_m)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[1]);

  Matrix<double,Dynamic,Dynamic> L_m(1,1);
  Matrix<double,Dynamic,Dynamic> L_p(1,1);
  L_p(0) = L(0) + epsilon;
  L_m(0) = L(0) - epsilon;
  grad_diff 
    =  (multi_gp_cholesky_log(y,L_p,w) - multi_gp_cholesky_log(y,L_m,w)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[2]);
}


struct multi_gp_cholesky_fun {
  const int K_, N_;

  multi_gp_cholesky_fun(int K, int N) : K_(K), N_(N) { }

  template <typename T>
  T operator()(const std::vector<T>& x) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using stan::math::var;
    Matrix<T,Dynamic,Dynamic> y(K_,N_);
    Matrix<T,Dynamic,Dynamic> Sigma(N_,N_);
    Matrix<T,Dynamic,1> w(K_);

    int pos = 0;
    for (int j = 0; j < N_; ++j)
      for (int i = 0; i < K_; ++i)
        y(i,j) = x[pos++];
    for (int j = 0; j < N_; ++j) {
      for (int i = 0; i <= j; ++i) {
        Sigma(i,j) = x[pos++];
        Sigma(j,i) = Sigma(i,j);
      }
    }
    Matrix<T,Dynamic,Dynamic> L = Sigma.llt().matrixL();
    for (int i = 0; i < K_; ++i)
      w(i) = x[pos++];
    return stan::math::multi_gp_cholesky_log<false>(y,L,w);
  }
};

TEST(MultiGPCholesky, TestGradFunctional) {
  std::vector<double> x(3*2 + 3 + 3);
  // y
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = -3.0;

  x[3] = 0.0;
  x[4] = -2.0;
  x[5] = -3.0;

  // Sigma
  x[6] = 1;
  x[7] = -1;
  x[8] = 10;

  // w
  x[9] = 1;
  x[10] = 10;
  x[11] = 5;



  test_grad(multi_gp_cholesky_fun(3,2), x);

  std::vector<double> u(3);
  u[0] = 1.9;
  u[1] = 0.48;
  u[2] = 2.7;
  
  test_grad(multi_gp_cholesky_fun(1,1), u);
}
