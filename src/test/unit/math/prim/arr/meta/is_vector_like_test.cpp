#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST(is_vector_like, vector) {
  EXPECT_TRUE(stan::is_vector_like<std::vector<double> >::value);
}
