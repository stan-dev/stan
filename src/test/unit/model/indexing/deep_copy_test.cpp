#include <vector>
#include <stan/model/indexing/deep_copy.hpp>
#include <gtest/gtest.h>

// TEST(ModelDeepCopy, scalar) {
//   using stan::model::deep_copy;
//   EXPECT_FLOAT_EQ(2.3, deep_copy(2.3));
//   EXPECT_EQ(3, deep_copy(3));
// }

// TEST(ModelDeepCopy, stdVector) {
//   using stan::model::deep_copy;
//   std::vector<double> x;
//   x.push_back(2.3);
//   x.push_back(3.7);
//   x.push_back(-1.9);
//   std::vector<double> y = deep_copy(x);
//   EXPECT_FLOAT_EQ(3.7, y[1]);
//   EXPECT_FLOAT_EQ(3.7, x[1]);
//   y[1] = 7.2;
//   EXPECT_FLOAT_EQ(7.2, y[1]);
//   EXPECT_FLOAT_EQ(3.7, x[1]);
// }
