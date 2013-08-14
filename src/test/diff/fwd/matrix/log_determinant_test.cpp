#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/diff/fwd/matrix/log_determinant.hpp>
#include <stan/diff/fwd/fvar.hpp>

TEST(DiffFwdMatrix,log_determinant) {
  using stan::diff::matrix_fv;
  using stan::diff::fvar;
  using stan::diff::log_determinant;
  
  matrix_fv v(2,2);
  v << 0, 1, 2, 3;
  v(0,0).d_ = 1.0;
  v(0,1).d_ = 2.0;
  v(1,0).d_ = 2.0;
  v(1,1).d_ = 2.0;
  
  fvar<double> det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_);
  EXPECT_FLOAT_EQ(1.5, det.d_);
}

TEST(DiffFwdMatrix,log_deteriminant_exception) {
  using stan::diff::matrix_fv;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_fv(2,3)), std::domain_error);
}
