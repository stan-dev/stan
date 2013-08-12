#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/log_determinant.hpp>
#include <stan/agrad/fwd/fvar.hpp>

TEST(AgradFwdMatrix,log_determinant) {
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::log_determinant;
  
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

TEST(AgradFwdMatrix,log_deteriminant_exception) {
  using stan::agrad::matrix_fv;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_fv(2,3)), std::domain_error);
}
