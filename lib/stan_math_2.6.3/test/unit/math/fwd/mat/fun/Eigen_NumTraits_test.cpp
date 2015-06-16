#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixEigenNumTraits, insertion_operator) {
  using stan::math::fvar;
  std::stringstream output;

  Eigen::Matrix<fvar<double>, -1, -1> A(2, 2);
  A << 1, 2, 3, 4;
  output << A;
  EXPECT_EQ("1 2\n3 4", output.str());
}
