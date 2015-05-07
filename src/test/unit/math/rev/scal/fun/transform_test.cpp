#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <stan/math/rev/mat/fun/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log1p.hpp>
#include <stan/math/rev/scal/fun/log1m.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/tanh.hpp>
#include <stan/math/prim/mat/fun/factor_U.hpp>
#include <stan/math/prim/mat/fun/determinant.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/factor_cov_matrix.hpp>
#include <stan/math/prim/mat/fun/read_corr_L.hpp>
#include <stan/math/prim/mat/fun/read_corr_matrix.hpp>
#include <stan/math/prim/mat/fun/read_cov_L.hpp>
#include <stan/math/prim/mat/fun/read_cov_matrix.hpp>
#include <stan/math/prim/mat/fun/make_nu.hpp>
#include <stan/math/prim/scal/fun/identity_constrain.hpp>
#include <stan/math/prim/scal/fun/identity_free.hpp>
#include <stan/math/prim/scal/fun/positive_constrain.hpp>
#include <stan/math/prim/scal/fun/positive_free.hpp>
#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>
#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>
#include <stan/math/prim/scal/fun/lub_constrain.hpp>
#include <stan/math/prim/scal/fun/lub_free.hpp>
#include <stan/math/prim/scal/fun/prob_constrain.hpp>
#include <stan/math/prim/scal/fun/prob_free.hpp>
#include <stan/math/prim/scal/fun/corr_constrain.hpp>
#include <stan/math/prim/scal/fun/corr_free.hpp>
#include <stan/math/prim/mat/fun/unit_vector_constrain.hpp>
#include <stan/math/prim/mat/fun/unit_vector_free.hpp>
#include <stan/math/prim/mat/fun/simplex_constrain.hpp>
#include <stan/math/prim/mat/fun/simplex_free.hpp>
#include <stan/math/prim/mat/fun/ordered_constrain.hpp>
#include <stan/math/prim/mat/fun/ordered_free.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_constrain.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_free.hpp>
#include <stan/math/prim/mat/fun/corr_matrix_constrain.hpp>
#include <stan/math/prim/mat/fun/corr_matrix_free.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_constrain.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_free.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_constrain_lkj.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_free_lkj.hpp>
#include <test/unit/util.hpp>
#include <test/unit/math/rev/mat/fun/jacobian.hpp>

#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform,ordered_jacobian_ad) {
  using stan::math::var;
  using stan::math::ordered_constrain;
  using stan::math::determinant;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double,Dynamic,1> x(3);
  x << -12.0, 3.0, -1.9;
  double lp = 0.0;
  Matrix<double,Dynamic,1> y = ordered_constrain(x,lp);

  Matrix<var,Dynamic,1> xv(3);
  xv << -12.0, 3.0, -1.9;

  std::vector<var> xvec(3);
  for (int i = 0; i < 3; ++i)
    xvec[i] = xv[i];

  Matrix<var,Dynamic,1> yv = ordered_constrain(xv);


  EXPECT_EQ(y.size(), yv.size());
  for (int i = 0; i < y.size(); ++i)
    EXPECT_FLOAT_EQ(y(i),yv(i).val());

  std::vector<var> yvec(3);
  for (unsigned int i = 0; i < 3; ++i)
    yvec[i] = yv[i];

  std::vector<std::vector<double> > j;
  stan::math::jacobian(yvec,xvec,j);

  Matrix<double,Dynamic,Dynamic> J(3,3);
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 3; ++n)
      J(m,n) = j[m][n];
  
  double log_abs_jacobian_det = log(fabs(determinant(J)));
  EXPECT_FLOAT_EQ(log_abs_jacobian_det, lp);
}

TEST(prob_transform,positive_ordered_jacobian_ad) {
  using stan::math::var;
  using stan::math::positive_ordered_constrain;
  using stan::math::determinant;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double,Dynamic,1> x(3);
  x << -12.0, 3.0, -1.9;
  double lp = 0.0;
  Matrix<double,Dynamic,1> y = positive_ordered_constrain(x,lp);

  Matrix<var,Dynamic,1> xv(3);
  xv << -12.0, 3.0, -1.9;

  std::vector<var> xvec(3);
  for (int i = 0; i < 3; ++i)
    xvec[i] = xv[i];

  Matrix<var,Dynamic,1> yv = positive_ordered_constrain(xv);


  EXPECT_EQ(y.size(), yv.size());
  for (int i = 0; i < y.size(); ++i)
    EXPECT_FLOAT_EQ(y(i),yv(i).val());

  std::vector<var> yvec(3);
  for (unsigned int i = 0; i < 3; ++i)
    yvec[i] = yv[i];

  std::vector<std::vector<double> > j;
  stan::math::jacobian(yvec,xvec,j);

  Matrix<double,Dynamic,Dynamic> J(3,3);
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 3; ++n)
      J(m,n) = j[m][n];
  
  double log_abs_jacobian_det = log(fabs(determinant(J)));
  EXPECT_FLOAT_EQ(log_abs_jacobian_det, lp);
}

TEST(prob_transform,corr_matrix_jacobian) {
  using stan::math::var;
  using stan::math::determinant;
  using std::log;
  using std::fabs;

  int K = 4;
  int K_choose_2 = 6;
  Matrix<var,Dynamic,1> X(K_choose_2);
  X << 1.0, 2.0, -3.0, 1.7, 9.8, -1.2;
  std::vector<var> x;
  for (int i = 0; i < X.size(); ++i)
    x.push_back(X(i));
  var lp = 0.0;
  Matrix<var,Dynamic,Dynamic> Sigma = stan::math::corr_matrix_constrain(X,K,lp);
  std::vector<var> y;
  for (int m = 0; m < K; ++m)
    for (int n = 0; n < m; ++n)
      y.push_back(Sigma(m,n));
  EXPECT_EQ(K_choose_2, y.size());

  std::vector<std::vector<double> > j;
  stan::math::jacobian(y,x,j);

  Matrix<double,Dynamic,Dynamic> J(X.size(),X.size());
  for (int m = 0; m < J.rows(); ++m)
    for (int n = 0; n < J.cols(); ++n)
      J(m,n) = j[m][n];

  double log_abs_jacobian_det = log(fabs(determinant(J)));
  EXPECT_FLOAT_EQ(log_abs_jacobian_det,lp.val());
}


TEST(prob_transform,cov_matrix_jacobian) {
  using stan::math::var;
  using stan::math::determinant;
  using std::log;
  using std::fabs;

  int K = 4;
  //unsigned int K = 4;
  unsigned int K_choose_2 = 6;
  Matrix<var,Dynamic,1> X(K_choose_2 + K);
  X << 1.0, 2.0, -3.0, 1.7, 9.8, 
    -12.2, 0.4, 0.2, 1.2, 2.7;
  std::vector<var> x;
  for (int i = 0; i < X.size(); ++i)
    x.push_back(X(i));
  var lp = 0.0;
  Matrix<var,Dynamic,Dynamic> Sigma = stan::math::cov_matrix_constrain(X,K,lp);
  std::vector<var> y;
  for (int m = 0; m < K; ++m)
    for (int n = 0; n <= m; ++n)
      y.push_back(Sigma(m,n));

  std::vector<std::vector<double> > j;
  stan::math::jacobian(y,x,j);

  Matrix<double,Dynamic,Dynamic> J(10,10);
  for (int m = 0; m < 10; ++m)
    for (int n = 0; n < 10; ++n)
      J(m,n) = j[m][n];

  double log_abs_jacobian_det = log(fabs(determinant(J)));
  EXPECT_FLOAT_EQ(log_abs_jacobian_det,lp.val());
}

TEST(probTransform,simplex_jacobian) {
  using stan::math::var;
  using std::vector;
  var a = 2.0;
  var b = 3.0;
  var c = -1.0;
  
  Matrix<var,Dynamic,1> y(3);
  y << a, b, c;
  
  var lp(0);
  Matrix<var,Dynamic,1> x 
    = stan::math::simplex_constrain(y,lp);
  
  vector<var> indeps;
  indeps.push_back(a);
  indeps.push_back(b);
  indeps.push_back(c);

  vector<var> deps;
  deps.push_back(x(0));
  deps.push_back(x(1));
  deps.push_back(x(2));
  
  vector<vector<double> > jacobian;
  stan::math::jacobian(deps,indeps,jacobian);

  Matrix<double,Dynamic,Dynamic> J(3,3);
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 3; ++n)
      J(m,n) = jacobian[m][n];
  
  double det_J = J.determinant();
  double log_det_J = log(det_J);

  EXPECT_FLOAT_EQ(log_det_J, lp.val());
  
}

TEST(probTransform,unit_vector_jacobian) {
  using stan::math::var;
  using std::vector;
  var a = 2.0;
  var b = 3.0;
  var c = -1.0;
  
  Matrix<var,Dynamic,1> y(3);
  y << a, b, c;
  
  var lp(0);
  Matrix<var,Dynamic,1> x 
    = stan::math::unit_vector_constrain(y,lp);
  
  vector<var> indeps;
  indeps.push_back(a);
  indeps.push_back(b);
  indeps.push_back(c);

  vector<var> deps;
  deps.push_back(x(0));
  deps.push_back(x(1));
  deps.push_back(x(2));
  deps.push_back(x(3));
  
  vector<vector<double> > jacobian;
  stan::math::jacobian(deps,indeps,jacobian);

  Matrix<double,Dynamic,Dynamic> J(4,4);
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 3; ++n) {
      J(m,n) = jacobian[m][n];
    }
    J(m,3) = x(m).val(); 
  }
  
  double det_J = J.determinant();
  double log_det_J = log(fabs(det_J));

  EXPECT_FLOAT_EQ(log_det_J, lp.val()) << "J = " << J << std::endl << "det_J = " << det_J;
  
}


void 
test_cholesky_correlation_jacobian(const Eigen::Matrix<stan::math::var,
                                                       Eigen::Dynamic,1>& y,
                                   int K) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::cholesky_corr_constrain;

  int K_choose_2 = (K * (K - 1)) / 2;

  vector<var> indeps;
  for (int i = 0; i < y.size(); ++i)
    indeps.push_back(y(i));

  var lp = 0;
  Matrix<var,Dynamic,Dynamic> x
    = cholesky_corr_constrain(y,K,lp);

  vector<var> deps;
  for (int i = 1; i < K; ++i)
    for (int j = 0; j < i; ++j)
      deps.push_back(x(i,j));
  
  vector<vector<double> > jacobian;
  stan::math::jacobian(deps,indeps,jacobian);

  Matrix<double,Dynamic,Dynamic> J(K_choose_2,K_choose_2);
  for (int m = 0; m < K_choose_2; ++m)
    for (int n = 0; n < K_choose_2; ++n)
      J(m,n) = jacobian[m][n];

  
  double det_J = J.determinant();
  double log_det_J = log(fabs(det_J));

  EXPECT_FLOAT_EQ(log_det_J, lp.val()) << "J = " << J << std::endl << "det_J = " << det_J;
  
}

TEST(probTransform,choleskyCorrJacobian) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  // K = 1; (K choose 2) = 0
  Matrix<var,Dynamic,1> y1;
  EXPECT_EQ(0,y1.size());
  test_cholesky_correlation_jacobian(y1,1);

  // K = 2; (K choose 2) = 1
  Matrix<var,Dynamic,1> y2(1);
  y2 << -1.7;
  test_cholesky_correlation_jacobian(y2,2);

  // K = 3; (K choose 2) = 3
  Matrix<var,Dynamic,1> y3(3);
  y3 << -1.7, 2.9, 0.01;
  test_cholesky_correlation_jacobian(y3,3);

  // K = 4;  (K choose 2) = 6
  Matrix<var,Dynamic,1> y4(6);
  y4 << 1.0, 2.0, -3.0, 1.5, 0.2, 2.0;
  test_cholesky_correlation_jacobian(y4,4);
}
