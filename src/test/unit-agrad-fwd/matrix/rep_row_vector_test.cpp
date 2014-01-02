#include <gtest/gtest.h>
#include <stan/math/rep_row_vector.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::var;
TEST(AgradFwdMatrixRepRowVector,fd_rowvector) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_fd;
  using stan::agrad::fvar;
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
  using stan::agrad::row_vector_fd;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
TEST(AgradFwdMatrixRepRowVector,fv_rowvector) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  row_vector_fv output;
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

TEST(AgradFwdMatrixRepRowVector,fv_rowvector_exception) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
TEST(AgradFwdMatrixRepRowVector,ffd_rowvector) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;
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
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
TEST(AgradFwdMatrixRepRowVector,ffv_rowvector) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  fvar<fvar<var> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  row_vector_ffv output;
  output = rep_row_vector(a, 4);

  EXPECT_EQ(3,output(0).val_.val().val());
  EXPECT_EQ(3,output(1).val_.val().val());
  EXPECT_EQ(3,output(2).val_.val().val());
  EXPECT_EQ(3,output(3).val_.val().val());
  EXPECT_EQ(2,output(0).d_.val().val());
  EXPECT_EQ(2,output(1).d_.val().val());
  EXPECT_EQ(2,output(2).d_.val().val());
  EXPECT_EQ(2,output(3).d_.val().val());
}

TEST(AgradFwdMatrixRepRowVector,ffv_rowvector_exception) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  fvar<fvar<var> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
