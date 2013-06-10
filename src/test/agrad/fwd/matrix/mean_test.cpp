#include <stan/math/matrix/mean.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix, mean_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;

  vector_d d1(3);
  vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrix, mean_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d;
  vector_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrix, mean_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrix, mean_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d;
  row_vector_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrix, mean_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;

  matrix_d d1(3,1);
  matrix_fv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<double> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(0.0, output.d_);
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_);
  EXPECT_FLOAT_EQ(1.0, output.d_);
}
TEST(AgradFwdMatrix, mean_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
 
  matrix_d d;
  matrix_fv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFwdMatrix, meanStdVector) {
  using stan::math::mean;
  using stan::agrad::fvar;

  std::vector<fvar<double> > x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_);
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_);

  std::vector<fvar<double> > y;
  fvar<double> a = 1.0;
  a.d_ = 1.0;
  fvar<double> b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<double> f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_);
  EXPECT_FLOAT_EQ(1.0, f.d_);
}
TEST(AgradFvarVarMatrix, mean_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d d1(3);
  vector_fvv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFvarVarMatrix, mean_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_fvv;

  vector_d d;
  vector_fvv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFvarVarMatrix, mean_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d d1(3);
  row_vector_fvv v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0).d_ = 1.0;
   v1(1).d_ = 1.0;
   v1(2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFvarVarMatrix, mean_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fvv;

  row_vector_d d;
  row_vector_fvv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFvarVarMatrix, mean_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d d1(3,1);
  matrix_fvv v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
   v1(0,0).d_ = 1.0;
   v1(0,1).d_ = 1.0;
   v1(0,2).d_ = 1.0;
  
  fvar<var> output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFvarVarMatrix, mean_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;
 
  matrix_d d;
  matrix_fvv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFvarVarMatrix, meanStdVector) {
  using stan::math::mean;
  using stan::agrad::fvar;
  using stan::agrad::var;

  std::vector<fvar<var> > x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_.val());
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_.val());

  std::vector<fvar<var> > y;
  fvar<var> a = 1.0;
  a.d_ = 1.0;
  fvar<var> b = 2.0;
  b.d_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<var> f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_.val());
  EXPECT_FLOAT_EQ(1.0, f.d_.val());
}
TEST(AgradFvarFvarMatrix, mean_vector) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;

  vector_d d1(3);
  vector_ffv v1(3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFvarFvarMatrix, mean_vector_exception) {
  using stan::math::mean;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_d d;
  vector_ffv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFvarFvarMatrix, mean_rowvector) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  
  fvar<fvar<double> > output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFvarFvarMatrix, mean_rowvector_exception) {
  using stan::math::mean;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_d d;
  row_vector_ffv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFvarFvarMatrix, mean_matrix) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

  matrix_d d1(3,1);
  matrix_ffv v1(1,3);
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;

  fvar<fvar<double> > output;
  output = mean(d1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(0.0, output.d_.val());
                   
  output = mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val_.val());
  EXPECT_FLOAT_EQ(1.0, output.d_.val());
}
TEST(AgradFvarFvarMatrix, mean_matrix_exception) {
  using stan::math::mean;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
 
  matrix_d d;
  matrix_ffv v;
  EXPECT_THROW(mean(d), std::domain_error);
  EXPECT_THROW(mean(v), std::domain_error);
}
TEST(AgradFvarFvarMatrix, meanStdVector) {
  using stan::math::mean;
  using stan::agrad::fvar;

  std::vector<fvar<fvar<double> > > x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val_.val());
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val_.val());

  std::vector<fvar<fvar<double> > > y;
  fvar<fvar<double> > a,b;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  y.push_back(a);
  y.push_back(b);
  fvar<fvar<double> > f = mean(y);

  EXPECT_FLOAT_EQ(1.5, f.val_.val());
  EXPECT_FLOAT_EQ(1.0, f.d_.val());
}
