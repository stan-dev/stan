#include <stan/math.hpp>
#include <gtest/gtest.h>

using stan::agrad::var;
using std::vector;
using Eigen::Matrix;
using stan::math::value_of_rec;
using stan::agrad::value_of_rec;

TEST(Math, includes_in_correct_order) {
  // quick test to make sure this compiles
  var a(1.0);
  value_of_rec(a);

  Matrix<var, -1, -1> b(2, 1);
  b(0) = 2.0;
  b(1) = 3.0;
  value_of_rec(b);
}
