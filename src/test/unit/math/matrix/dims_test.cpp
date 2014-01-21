#include <stan/math/matrix/dims.hpp>
#include <stan/math/matrix.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, dims) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::dims;

  double x1;
  vector<int> dims1 = dims(x1);
  EXPECT_EQ(0U,dims1.size());

  int x2;
  vector<int> dims2 = dims(x2);
  EXPECT_EQ(0U,dims2.size());

  vector<double> x3;
  x3.push_back(-32.1); x3.push_back(17.9);
  vector<int> dims3 = dims(x3);
  EXPECT_EQ(1U,dims3.size());
  EXPECT_EQ(2,dims3[0]);

  vector<vector<double> > x4;
  x4.push_back(x3);  x4.push_back(x3);   x4.push_back(x3);
  vector<int> dims4 = dims(x4);
  EXPECT_EQ(2U,dims4.size());
  EXPECT_EQ(3,dims4[0]);
  EXPECT_EQ(2,dims4[1]);

  Matrix<double,Dynamic,Dynamic> x5(7,8);
  vector<int> dims5 = dims(x5);
  EXPECT_EQ(2U,dims5.size());
  EXPECT_EQ(7,dims5[0]);
  EXPECT_EQ(8,dims5[1]);

  Matrix<double,Dynamic,1> x6(17);
  vector<int> dims6 = dims(x6);
  EXPECT_EQ(2U,dims6.size());
  EXPECT_EQ(17,dims6[0]);
  EXPECT_EQ(1,dims6[1]);

  Matrix<double,1,Dynamic> x7(17);
  vector<int> dims7 = dims(x7);
  EXPECT_EQ(2U,dims7.size());
  EXPECT_EQ(1,dims7[0]);
  EXPECT_EQ(17,dims7[1]);

  vector<Matrix<double,Dynamic,Dynamic> > x8;
  x8.push_back(x5);  x8.push_back(x5);
  vector<int> dims8 = dims(x8);
  EXPECT_EQ(3U,dims8.size());
  EXPECT_EQ(2,dims8[0]);
  EXPECT_EQ(7,dims8[1]);
  EXPECT_EQ(8,dims8[2]);
}
