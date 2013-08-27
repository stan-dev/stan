#include <stan/diff/rev/matrix/log_determinant.hpp>
#include <gtest/gtest.h>
#include <test/diff/util.hpp>
#include <stan/math/matrix/determinant.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/diff/rev/fabs.hpp>
#include <stan/diff/rev/log.hpp>

TEST(DiffRevMatrix,log_determinant_diff) {
  using stan::diff::matrix_v;
  using stan::math::determinant;
  using stan::diff::fabs;
  using stan::diff::log;

  // expected from auto-diff/Eigen
  AVEC x1 = createAVEC(0,1,2,3);
  matrix_v v1(2,2);
  v1 << x1[0], x1[1], x1[2], x1[3];
  AVAR det1 = log(fabs(v1.determinant()));
  std::vector<double> g1;
  det1.grad(x1,g1);
  
  // optimized in diff::matrix
  AVEC x2 = createAVEC(0,1,2,3);
  matrix_v v2(2,2);
  v2 << x2[0], x2[1], x2[2], x2[3];
  AVAR det2 = log_determinant(v2);
  std::vector<double> g2;
  det2.grad(x2,g2);

  EXPECT_FLOAT_EQ(det1.val(), det2.val());
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i],g2[i]);
}

TEST(DiffRevMatrix,log_determinant) {
  using stan::diff::matrix_v;
  using stan::math::log_determinant;
  
  matrix_v v(2,2);
  v << 0, 1, 2, 3;
  
  AVAR det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val());
}

TEST(DiffRevMatrix,log_deteriminant_exception) {
  using stan::diff::matrix_v;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_v(2,3)), std::domain_error);
}

TEST(DiffRevMatrix,log_determinant_grad) {
  using stan::diff::matrix_v;
  using stan::math::log_determinant;
  
  matrix_v X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, c, d;
  
  AVEC x = createAVEC(a,b,c,d);
  
  AVAR f = log_determinant(X);
  
  // det = ad - bc
  EXPECT_NEAR(0.0,f.val(),1E-12);
  
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-7.0,g[0]);
  EXPECT_FLOAT_EQ(5.0,g[1]);
  EXPECT_FLOAT_EQ(3.0,g[2]);
  EXPECT_FLOAT_EQ(-2.0,g[3]);
}
