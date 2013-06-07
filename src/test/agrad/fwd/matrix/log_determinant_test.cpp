#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/log_determinant.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/var.hpp>

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
TEST(AgradFwdFvarVarMatrix,log_determinant) {
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_determinant;
  
  fvar<var> a(0.0,1.0);
  fvar<var> b(1.0,2.0);
  fvar<var> c(2.0,2.0);
  fvar<var> d(3.0,2.0);

  matrix_fvv v(2,2);
  v << a,b,c,d;
  
  fvar<var> det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_.val());
  EXPECT_FLOAT_EQ(1.5, det.d_.val());
}

TEST(AgradFwdFvarVarMatrix,log_deteriminant_exception) {
  using stan::agrad::matrix_fvv;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_fvv(2,3)), std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix,log_determinant) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::log_determinant;
  
  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 0.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 1.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 3.0;
  d.d_.val_ = 2.0; 

  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  fvar<fvar<double> > det;
  det = log_determinant(v);
  EXPECT_FLOAT_EQ(std::log(2.0), det.val_.val());
  EXPECT_FLOAT_EQ(1.5, det.d_.val());
}

TEST(AgradFwdFvarFvarMatrix,log_deteriminant_exception) {
  using stan::agrad::matrix_ffv;
  using stan::math::log_determinant;
  
  EXPECT_THROW(log_determinant(matrix_ffv(2,3)), std::domain_error);
}
