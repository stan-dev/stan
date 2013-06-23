#include <gtest/gtest.h>
#include <stan/math/rep_array.hpp>

TEST(MathMatrix,rep_array) {
  using stan::math::rep_array;
  std::vector<double> x = rep_array(2.0, 3);
  EXPECT_EQ(3U,x.size());
  for (size_t i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(2.0, x[i]);

  EXPECT_THROW(rep_array(2.0,-2), std::domain_error);
}
TEST(MathMatrix,rep_array2D) {
  using stan::math::rep_array;
  using std::vector;
  vector<vector<double> > x = rep_array(2.0, 3, 4);
  EXPECT_EQ(3U,x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(4U,x[i].size());
    for (size_t j = 0; j < x[i].size(); ++j)
      EXPECT_FLOAT_EQ(2.0, x[i][j]);
  }
  EXPECT_THROW(rep_array(2.0,-2,3), std::domain_error);
  EXPECT_THROW(rep_array(2.0,2,-3), std::domain_error);
}
TEST(MathMatrix,rep_array3D) {
  using stan::math::rep_array;
  using std::vector;
  vector<vector<vector<int> > > x = rep_array(13, 3, 4, 5);
  EXPECT_EQ(3U,x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(4U,x[i].size());
    for (size_t j = 0; j < x[i].size(); ++j) {
      EXPECT_EQ(5U,x[i][j].size());
      for (size_t k = 0; k < x[i][j].size(); ++k)
        EXPECT_EQ(13, x[i][j][k]);
    }
  }
  EXPECT_THROW(rep_array(2.0,-2,3,4), std::domain_error);
  EXPECT_THROW(rep_array(2.0,2,-3,4), std::domain_error);
  EXPECT_THROW(rep_array(2.0,2,3,-4), std::domain_error);
}
