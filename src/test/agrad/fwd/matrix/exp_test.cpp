#include <stan/math/matrix/exp.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/exp.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix, exp_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;

  matrix_d expected_output(2,2);
  matrix_fd mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
   mv(0).d_ = 2.0;
   mv(1).d_ = 2.0;
   mv(2).d_ = 2.0;
   mv(3).d_ = 2.0;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0,0).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(0,1).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(1,0).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(1,1).d_);
}
TEST(AgradFwdMatrix, exp_vector) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d expected_output(4);
  vector_fd mv(4), output;

  mv << 1, 2, 3, 4;
   mv(0).d_ = 2.0;
   mv(1).d_ = 2.0;
   mv(2).d_ = 2.0;
   mv(3).d_ = 2.0;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_);
}
TEST(AgradFwdMatrix, exp_rowvector) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d expected_output(4);
  row_vector_fd mv(4), output;

  mv << 1, 2, 3, 4;
   mv(0).d_ = 2.0;
   mv(1).d_ = 2.0;
   mv(2).d_ = 2.0;
   mv(3).d_ = 2.0;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_);
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_);
}
TEST(AgradFvarVarMatrix, exp_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d expected_output(2,2);
  matrix_fv mv(2,2), output;
  int i,j;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0,0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(0,1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(1,0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(1,1).d_.val());
}
TEST(AgradFvarVarMatrix, exp_vector) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d expected_output(4);
  vector_fv mv(4), output;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_.val());
}
TEST(AgradFvarVarMatrix, exp_rowvector) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d expected_output(4);
  row_vector_fv mv(4), output;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_.val());
}
TEST(AgradFvarFvarMatrix, exp_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::fvar;

  matrix_d expected_output(2,2);
  matrix_ffd mv(2,2), output;
  int i,j;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 2.0; 

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0,0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(0,1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(1,0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(1,1).d_.val());
}
TEST(AgradFvarFvarMatrix, exp_vector) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  vector_d expected_output(4);
  vector_ffd mv(4), output;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 2.0; 

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_.val());
}
TEST(AgradFvarFvarMatrix, exp_rowvector) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  row_vector_d expected_output(4);
  row_vector_ffd mv(4), output;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 2.0; 

  mv << a,b,c,d;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(expected_output(i), output(i).val_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(1), output(0).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(2), output(1).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(3), output(2).d_.val());
  EXPECT_FLOAT_EQ(2.0 * std::exp(4), output(3).d_.val());
}
