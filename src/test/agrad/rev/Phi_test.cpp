#include <stan/agrad/rev/Phi.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev, Phi) {
  using stan::agrad::var;
  using std::exp;

  std::vector<double> y_values;
  y_values.push_back(0.0);
  y_values.push_back(0.9);
  y_values.push_back(-5.0);

  // d/dy = exp(normal_log(value_of(y), 0.0, 1.0))
  std::vector<double> dy_values;
  dy_values.push_back(0.3989423);
  dy_values.push_back(0.2660852);
  dy_values.push_back(1.4867195e-06);

  for (size_t i = 0; i < y_values.size(); i++) {
    var y, phi_y;
    AVEC x;
    VEC dy;
    y = y_values[i];
    phi_y = stan::agrad::Phi(y);
    x = createAVEC(y);
    phi_y.grad(x,dy);
    EXPECT_FLOAT_EQ(stan::math::Phi(y.val()), phi_y.val());
    EXPECT_FLOAT_EQ(dy_values[i], dy[0])
      << "y = " << y;
  }
}
