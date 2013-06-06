#include <stan/math/matrix/max.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

using stan::agrad::fvar;
TEST(AgradFwdMatrix, max_vector) {
  using stan::math::max;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, max_vector_exception) {
  using stan::math::max;
  using stan::agrad::vector_fv;

  vector_fv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
TEST(AgradFwdMatrix, max_rowvector) {
  using stan::math::max;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, max_rowvector_exception) {
  using stan::math::max;
  using stan::agrad::row_vector_fv;

  row_vector_fv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
TEST(AgradFwdMatrix, max_matrix) {
  using stan::math::max;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d d1(3,1);
  matrix_fv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<double> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(0, output.d_);
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_);
  EXPECT_FLOAT_EQ(1, output.d_);
}
TEST(AgradFwdMatrix, max_matrix_exception) {
  using stan::math::max;
  using stan::agrad::matrix_fv;
  
  matrix_fv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_);
  EXPECT_EQ(0, max(v).d_);
}
TEST(AgradFwdFvarVarMatrix, max_vector) {
  using stan::math::max;
  using stan::math::vector_d;
  using stan::agrad::vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_fvv v1(3);
  
  fvar<var> a(100.0,1.0);
  fvar<var> b(0.0,1.0);
  fvar<var> c(-3.0,1.0);

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<var> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdFvarVarMatrix, max_vector_exception) {
  using stan::math::max;
  using stan::agrad::vector_fvv;

  vector_fvv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
TEST(AgradFwdFvarVarMatrix, max_rowvector) {
  using stan::math::max;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_fvv v1(3);
  
  fvar<var> a(100.0,1.0);
  fvar<var> b(0.0,1.0);
  fvar<var> c(-3.0,1.0);

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<var> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdFvarVarMatrix, max_rowvector_exception) {
  using stan::math::max;
  using stan::agrad::row_vector_fvv;

  row_vector_fvv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
TEST(AgradFwdFvarVarMatrix, max_matrix) {
  using stan::math::max;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_fvv v1(1,3);
  
  fvar<var> a(100.0,1.0);
  fvar<var> b(0.0,1.0);
  fvar<var> c(-3.0,1.0);

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<var> output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdFvarVarMatrix, max_matrix_exception) {
  using stan::math::max;
  using stan::agrad::matrix_fvv;
  
  matrix_fvv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, max_vector) {
  using stan::math::max;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;

  vector_d d1(3);
  vector_ffv v1(3);
  
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 0.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -3.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdFvarFvarMatrix, max_vector_exception) {
  using stan::math::max;
  using stan::agrad::vector_ffv;

  vector_ffv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, max_rowvector) {
  using stan::math::max;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 0.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -3.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdFvarFvarMatrix, max_rowvector_exception) {
  using stan::math::max;
  using stan::agrad::row_vector_ffv;

  row_vector_ffv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
TEST(AgradFwdFvarFvarMatrix, max_matrix) {
  using stan::math::max;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 0.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = -3.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = max(d1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(0, output.d_.val());
                   
  output = max(v1);
  EXPECT_FLOAT_EQ(100, output.val_.val());
  EXPECT_FLOAT_EQ(1, output.d_.val());
}
TEST(AgradFwdFvarFvarMatrix, max_matrix_exception) {
  using stan::math::max;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv v;
  EXPECT_EQ(-std::numeric_limits<double>::infinity(), max(v).val_.val());
  EXPECT_EQ(0, max(v).d_.val());
}
