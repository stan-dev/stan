#include <stan/math/matrix/dot_self.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, dot_self) {
  using stan::math::dot_self;

  Eigen::Matrix<double,Eigen::Dynamic,1> v1(1);
  v1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(v1),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,1> v2(2);
  v2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(v2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,1> v3(3);
  v3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(v3),1E-12);

  Eigen::Matrix<double,1,Eigen::Dynamic> rv1(1);
  rv1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(rv1),1E-12);
  Eigen::Matrix<double,1,Eigen::Dynamic> rv2(2);
  rv2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(rv2),1E-12);
  Eigen::Matrix<double,1,Eigen::Dynamic> rv3(3);
  rv3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(rv3),1E-12);

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(m1),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m2(2,1);
  m2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(m2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m3(3,1);
  m3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(m3),1E-12);

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mm2(1,2);
  mm2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(mm2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mm3(1,3);
  mm3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(mm3),1E-12);
}
