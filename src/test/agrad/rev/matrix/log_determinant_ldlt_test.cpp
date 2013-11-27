#include <stan/agrad/rev/matrix/log_determinant_ldlt.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix/log_determinant_spd.hpp>
#include <test/agrad/util.hpp>
#include <stan/math/matrix/determinant.hpp>
#include <stan/math/matrix/log_determinant_spd.hpp>
#include <stan/agrad/rev/fabs.hpp>
#include <stan/agrad/rev/log.hpp>

TEST(AgradRevMatrix,log_determinant_ldlt_diff) {
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
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_v;
  AVEC x2 = createAVEC(2,1,1,3);
  matrix_v v2(2,2);
  v2 << x2[0], x2[1], x2[2], x2[3];
  ldlt_v.compute(v2);
  AVAR det2 = log_determinant_ldlt(ldlt_v);
  std::vector<double> g2;
  det2.grad(x2,g2);

  EXPECT_FLOAT_EQ(det1.val(), det2.val());
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i],g2[i]);
}

TEST(AgradRevMatrix,log_determinant_ldlt) {
  using stan::agrad::matrix_v;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_v;
  
  matrix_v v(2,2);
  v << 1, 0, 0, 3;
  ldlt_v.compute(v);
  
  AVAR det;
  det = log_determinant_ldlt(ldlt_v);
  EXPECT_FLOAT_EQ(std::log(3.0), det.val());
}

