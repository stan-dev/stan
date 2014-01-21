#include <stan/agrad/rev/matrix/log_determinant_spd.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/determinant.hpp>
#include <stan/math/matrix/log_determinant_spd.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <stan/agrad/rev/functions/log.hpp>

TEST(AgradRevMatrix,log_determinant_spd_diff) {
  using stan::agrad::matrix_v;
  using stan::math::determinant;
  using stan::agrad::fabs;
  using stan::agrad::log;

  // expected from auto-diff/Eigen
  AVEC x1 = createAVEC(2,1,1,3);
  matrix_v v1(2,2);
  v1 << x1[0], x1[1], x1[2], x1[3];
  AVAR det1 = log(fabs(v1.determinant()));
  std::vector<double> g1;
  det1.grad(x1,g1);
  
  // optimized in agrad::matrix
  AVEC x2 = createAVEC(2,1,1,3);
  matrix_v v2(2,2);
  v2 << x2[0], x2[1], x2[2], x2[3];
  AVAR det2 = log_determinant_spd(v2);
  std::vector<double> g2;
  det2.grad(x2,g2);

  EXPECT_FLOAT_EQ(det1.val(), det2.val());
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i],g2[i]);
}

TEST(AgradRevMatrix,log_determinant_spd) {
  using stan::agrad::matrix_v;
  using stan::math::log_determinant_spd;
  
  matrix_v v(2,2);
  v << 1, 0, 0, 3;
  
  AVAR det;
  det = log_determinant_spd(v);
  EXPECT_FLOAT_EQ(std::log(3.0), det.val());
}
#if 0
TEST(AgradRevMatrix,log_deteriminant_exception) {
  using stan::agrad::matrix_v;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_v(2,3)), std::domain_error);
}

TEST(AgradRevMatrix,log_determinant_grad) {
  using stan::agrad::matrix_v;
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
#endif

