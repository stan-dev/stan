#include <stan/math/prim/mat/fun/exp.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixExp, fd_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;

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
TEST(AgradFwdMatrixExp, fd_vector) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::math::vector_fd;

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
TEST(AgradFwdMatrixExp, fd_rowvector) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

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
TEST(AgradFwdMatrixExp, ffd_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::fvar;

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
TEST(AgradFwdMatrixExp, ffd_vector) {
  using stan::math::exp;
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

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
TEST(AgradFwdMatrixExp, ffd_rowvector) {
  using stan::math::exp;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

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
