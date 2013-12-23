#include <stan/agrad/rev/jacobian.hpp>
#include <stan/agrad/rev/operators/operator_addition.hpp>
#include <stan/agrad/rev/operators/operator_multiplication.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,jacobian) {
  AVAR x1 = 2.0;
  AVAR x2 = 3.0;
  
  AVAR y1 = x1 * x2;
  AVAR y2 = x1 + x2;
  AVAR y3 = 17.0 * x1;

  AVEC x = createAVEC(x1,x2);
  AVEC y = createAVEC(y1,y2,y3);

  std::vector<std::vector<double> > J;
  jacobian(y,x,J);

  EXPECT_FLOAT_EQ(3.0,J[0][0]); // dy1/dx1
  EXPECT_FLOAT_EQ(2.0,J[0][1]); // dy1/dx2

  EXPECT_FLOAT_EQ(1.0,J[1][0]); // dy2/dx1
  EXPECT_FLOAT_EQ(1.0,J[1][1]); // dy2/dx2

  EXPECT_FLOAT_EQ(17.0,J[2][0]); // dy2/dx1
  EXPECT_FLOAT_EQ(0.0,J[2][1]); // dy2/dx2
}
