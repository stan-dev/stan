#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/rep_row_vector.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixRepRowVector,fd_rowvector) {
  using stan::math::rep_row_vector;
  using stan::math::row_vector_fd;
  using stan::math::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  row_vector_fd output;
  output = rep_row_vector(a, 4);

  EXPECT_EQ(3,output(0).val_);
  EXPECT_EQ(3,output(1).val_);
  EXPECT_EQ(3,output(2).val_);
  EXPECT_EQ(3,output(3).val_);
  EXPECT_EQ(2,output(0).d_);
  EXPECT_EQ(2,output(1).d_);
  EXPECT_EQ(2,output(2).d_);
  EXPECT_EQ(2,output(3).d_);
}

TEST(AgradFwdMatrixRepRowVector,fd_rowvector_exception) {
  using stan::math::rep_row_vector;
  using stan::math::row_vector_fd;
  using stan::math::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
TEST(AgradFwdMatrixRepRowVector,ffd_rowvector) {
  using stan::math::rep_row_vector;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  row_vector_ffd output;
  output = rep_row_vector(a, 4);

  EXPECT_EQ(3,output(0).val_.val());
  EXPECT_EQ(3,output(1).val_.val());
  EXPECT_EQ(3,output(2).val_.val());
  EXPECT_EQ(3,output(3).val_.val());
  EXPECT_EQ(2,output(0).d_.val());
  EXPECT_EQ(2,output(1).d_.val());
  EXPECT_EQ(2,output(2).d_.val());
  EXPECT_EQ(2,output(3).d_.val());
}

TEST(AgradFwdMatrixRepRowVector,ffd_rowvector_exception) {
  using stan::math::rep_row_vector;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
