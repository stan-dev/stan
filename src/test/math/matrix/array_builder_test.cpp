#include <stan/math/matrix/array_builder.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,arrayBuilder) {
  using std::vector;
  using stan::math::array_builder;

  EXPECT_EQ(0U, array_builder<double>().array().size());

  vector<double> x
    = array_builder<double>()
    .add(1)
    .add(3)
    .add(2)
    .array();
  EXPECT_EQ(3U,x.size());
  EXPECT_FLOAT_EQ(1.0, x[0]);
  EXPECT_FLOAT_EQ(3.0, x[1]);
  EXPECT_FLOAT_EQ(2.0, x[2]);

  vector<vector<int> > xx
    = array_builder<vector<int> >()
    .add(array_builder<int>().add(1).add(2).array())
    .add(array_builder<int>().add(3).add(4).array())
    .add(array_builder<int>().add(5).add(6).array())
    .array();

  EXPECT_EQ(3U,xx.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(2U,xx[i].size());
  EXPECT_EQ(1,xx[0][0]);
  EXPECT_EQ(2,xx[0][1]);
  EXPECT_EQ(3,xx[1][0]);
  EXPECT_EQ(4,xx[1][1]);
  EXPECT_EQ(5,xx[2][0]);
  EXPECT_EQ(6,xx[2][1]);
}

