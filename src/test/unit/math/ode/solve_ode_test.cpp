#include <gtest/gtest.h>
#include <stan/math/ode/solve_ode.hpp>
#include <stan/math/matrix/Eigen.hpp>

TEST(MathODE,solve_ode_to_std_vec) {
  using stan::math::to_std_vector;

  std::vector<double> v;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m(2,2);
  m << 1,2,3,4;

  to_std_vector(m, v);

  EXPECT_FLOAT_EQ(1, v[0]);
  EXPECT_FLOAT_EQ(3, v[1]);
  EXPECT_FLOAT_EQ(2, v[2]);
  EXPECT_FLOAT_EQ(4, v[3]);
}
TEST(MathODE,solve_ode_to_eigen_vec) {
}
TEST(MathODE,solve_ode_ode_system) {
}
TEST(MathODE,solve_ode_observer) {
}
TEST(MathODE,solve_ode_solve_ode) {
}
