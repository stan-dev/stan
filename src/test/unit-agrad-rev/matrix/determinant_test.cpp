#include <stan/agrad/rev/matrix/determinant.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/determinant.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <stan/agrad/rev/functions/log.hpp>

TEST(AgradRevMatrix,determinant) {
  using stan::agrad::matrix_v;
  using stan::math::determinant;

  // expected from auto-diff/Eigen
  AVEC x1 = createAVEC(0,1,2,3);
  matrix_v v1(2,2);
  v1 << x1[0], x1[1], x1[2], x1[3];
  AVAR det1 = v1.determinant();
  std::vector<double> g1;
  det1.grad(x1,g1);
  
  // optimized in agrad::matrix
  AVEC x2 = createAVEC(0,1,2,3);
  matrix_v v2(2,2);
  v2 << x2[0], x2[1], x2[2], x2[3];
  AVAR det2 = determinant(v2);
  std::vector<double> g2;
  det2.grad(x2,g2);

  EXPECT_FLOAT_EQ(det1.val(), det2.val());
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i],g2[i]);
}
TEST(AgradRevMatrix,deteriminant_exception) {
  using stan::agrad::matrix_v;
  using stan::math::determinant;

  EXPECT_THROW(determinant(matrix_v(2,3)), std::domain_error);
}
TEST(AgradRevMatrix,determinant_grad) {
  using stan::agrad::matrix_v;
  using stan::math::determinant;
  
  matrix_v X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, c, d;

  AVEC x = createAVEC(a,b,c,d);

  AVAR f = determinant(X);

  // det = ad - bc
  EXPECT_FLOAT_EQ(-1.0,f.val());

  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(7.0,g[0]);
  EXPECT_FLOAT_EQ(-5.0,g[1]);
  EXPECT_FLOAT_EQ(-3.0,g[2]);
  EXPECT_FLOAT_EQ(2.0,g[3]);
}
TEST(AgradRevMatrix,determinant3by3) {
  // just test it can handle it
  using stan::agrad::matrix_v;
  using stan::math::determinant;

  matrix_v Z(9,9);
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      Z(i,j) = i * j + 1;
  AVAR h = determinant(Z);
  h = h; // supresses set but not used warning
}
