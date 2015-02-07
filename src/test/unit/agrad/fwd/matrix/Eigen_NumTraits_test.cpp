#include <stan/agrad/fwd/matrix/Eigen_NumTraits.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixEigenNumTraits, insertion_operator) {
  using stan::agrad::fvar;
  std::stringstream output;

  Eigen::Matrix<fvar<double>, -1, -1> A(2, 2);
  A << 1, 2, 3, 4;
  output << A;
  EXPECT_EQ("  1:0   2:0\n  3:0   4:0", output.str());
}
