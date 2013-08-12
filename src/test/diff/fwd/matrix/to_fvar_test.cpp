#include <stan/agrad/fwd/matrix/to_fvar.hpp>
#include <stan/agrad/fvar.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrix,to_fvar_scalar) {
  using stan::agrad::fvar;
  double d = 5.0;
  fvar<double> v = 5.0;
   v.d_  = 1.0;
  fvar<double> fvar_x = stan::agrad::to_fvar(d);
  EXPECT_FLOAT_EQ(5.0, fvar_x.val_);
  EXPECT_FLOAT_EQ(0.0, fvar_x.d_);

  fvar_x = stan::agrad::to_fvar(v);
  EXPECT_FLOAT_EQ(5.0, fvar_x.val_);
  EXPECT_FLOAT_EQ(1.0, fvar_x.d_);
}
TEST(AgradFwdMatrix,to_fvar_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  matrix_d m_d(2,3);
  m_d << 0, 1, 2, 3, 4, 5;
  matrix_fv m_v = stan::agrad::to_fvar(m_d);

  EXPECT_EQ(2, m_v.rows());
  EXPECT_EQ(3, m_v.cols());
  for (int ii = 0; ii < 2; ii++) 
    for (int jj = 0; jj < 3; jj++) {
      EXPECT_FLOAT_EQ(ii*3 + jj, m_v(ii, jj).val_);
      EXPECT_FLOAT_EQ(0.0, m_v(ii, jj).d_);
    }
}
TEST(AgradFwdMatrix,to_fvar_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d(5);
  vector_fv v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
  
  vector_fv out = stan::agrad::to_fvar(d);
  EXPECT_FLOAT_EQ(1, out(0).val_);
  EXPECT_FLOAT_EQ(2, out(1).val_);
  EXPECT_FLOAT_EQ(3, out(2).val_);
  EXPECT_FLOAT_EQ(4, out(3).val_);
  EXPECT_FLOAT_EQ(5, out(4).val_);
  EXPECT_FLOAT_EQ(0, out(0).d_);
  EXPECT_FLOAT_EQ(0, out(1).d_);
  EXPECT_FLOAT_EQ(0, out(2).d_);
  EXPECT_FLOAT_EQ(0, out(3).d_);
  EXPECT_FLOAT_EQ(0, out(4).d_);

  out = stan::agrad::to_fvar(v);
  EXPECT_FLOAT_EQ(1, out(0).val_);
  EXPECT_FLOAT_EQ(2, out(1).val_);
  EXPECT_FLOAT_EQ(3, out(2).val_);
  EXPECT_FLOAT_EQ(4, out(3).val_);
  EXPECT_FLOAT_EQ(5, out(4).val_);  
  EXPECT_FLOAT_EQ(1, out(0).d_);
  EXPECT_FLOAT_EQ(1, out(1).d_);
  EXPECT_FLOAT_EQ(1, out(2).d_);
  EXPECT_FLOAT_EQ(1, out(3).d_);
  EXPECT_FLOAT_EQ(1, out(4).d_);
}
TEST(AgradFwdMatrix,to_fvar_rowvector) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d(5);
  row_vector_fv v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
   v(4).d_ = 1.0;
  
  row_vector_fv output = stan::agrad::to_fvar(d);
  EXPECT_FLOAT_EQ(1, output(0).val_);
  EXPECT_FLOAT_EQ(2, output(1).val_);
  EXPECT_FLOAT_EQ(3, output(2).val_);
  EXPECT_FLOAT_EQ(4, output(3).val_);
  EXPECT_FLOAT_EQ(5, output(4).val_);
  EXPECT_FLOAT_EQ(0, output(0).d_);
  EXPECT_FLOAT_EQ(0, output(1).d_);
  EXPECT_FLOAT_EQ(0, output(2).d_);
  EXPECT_FLOAT_EQ(0, output(3).d_);
  EXPECT_FLOAT_EQ(0, output(4).d_);

  output.resize(0);
  output = stan::agrad::to_fvar(v);
  EXPECT_FLOAT_EQ(1, output(0).val_);
  EXPECT_FLOAT_EQ(2, output(1).val_);
  EXPECT_FLOAT_EQ(3, output(2).val_);
  EXPECT_FLOAT_EQ(4, output(3).val_);
  EXPECT_FLOAT_EQ(5, output(4).val_);
  EXPECT_FLOAT_EQ(1, output(0).d_);
  EXPECT_FLOAT_EQ(1, output(1).d_);
  EXPECT_FLOAT_EQ(1, output(2).d_);
  EXPECT_FLOAT_EQ(1, output(3).d_);
  EXPECT_FLOAT_EQ(1, output(4).d_);
}
TEST(AgradFwdMatrix,to_fvar_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d val(3,3);
  matrix_d d(3,3);
  
  val <<1,2,3,4,5,6,7,8,9;
  d <<10,11,12,13,14,15,16,17,18;
  
  matrix_fv output = stan::agrad::to_fvar(val,d);
  EXPECT_FLOAT_EQ(1, output(0,0).val_);
  EXPECT_FLOAT_EQ(2, output(0,1).val_);
  EXPECT_FLOAT_EQ(3, output(0,2).val_);
  EXPECT_FLOAT_EQ(4, output(1,0).val_);
  EXPECT_FLOAT_EQ(5, output(1,1).val_);
  EXPECT_FLOAT_EQ(6, output(1,2).val_);
  EXPECT_FLOAT_EQ(7, output(2,0).val_);
  EXPECT_FLOAT_EQ(8, output(2,1).val_);
  EXPECT_FLOAT_EQ(9, output(2,2).val_);
  EXPECT_FLOAT_EQ(10, output(0,0).d_);
  EXPECT_FLOAT_EQ(11, output(0,1).d_);
  EXPECT_FLOAT_EQ(12, output(0,2).d_);
  EXPECT_FLOAT_EQ(13, output(1,0).d_);
  EXPECT_FLOAT_EQ(14, output(1,1).d_);
  EXPECT_FLOAT_EQ(15, output(1,2).d_);
  EXPECT_FLOAT_EQ(16, output(2,0).d_);
  EXPECT_FLOAT_EQ(17, output(2,1).d_);
  EXPECT_FLOAT_EQ(18, output(2,2).d_);

  matrix_d val1(4,4);
  EXPECT_THROW(stan::agrad::to_fvar(val1, d), std::domain_error);
}

