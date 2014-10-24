#include <gtest/gtest.h>
#include <stan/math/rep_vector.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::var;
TEST(AgradFwdMatrixRepVector,fd_vector) {
  using stan::math::rep_vector;
  using stan::agrad::vector_fd;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  vector_fd output;
  output = rep_vector(a, 4);

  EXPECT_EQ(3,output(0).val_);
  EXPECT_EQ(3,output(1).val_);
  EXPECT_EQ(3,output(2).val_);
  EXPECT_EQ(3,output(3).val_);
  EXPECT_EQ(2,output(0).d_);
  EXPECT_EQ(2,output(1).d_);
  EXPECT_EQ(2,output(2).d_);
  EXPECT_EQ(2,output(3).d_);
}

TEST(AgradFwdMatrixRepVector,fd_vector_exception) {
  using stan::math::rep_vector;
  using stan::agrad::vector_fd;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_vector(a,-2), std::domain_error);
}
TEST(AgradFwdMatrixRepVector,fv_vector) {
  using stan::math::rep_vector;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  vector_fv output;
  output = rep_vector(a, 4);

  EXPECT_EQ(3,output(0).val_.val());
  EXPECT_EQ(3,output(1).val_.val());
  EXPECT_EQ(3,output(2).val_.val());
  EXPECT_EQ(3,output(3).val_.val());
  EXPECT_EQ(2,output(0).d_.val());
  EXPECT_EQ(2,output(1).d_.val());
  EXPECT_EQ(2,output(2).d_.val());
  EXPECT_EQ(2,output(3).d_.val());
}

TEST(AgradFwdMatrixRepVector,fv_vector_exception) {
  using stan::math::rep_vector;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_vector(a,-2), std::domain_error);
}
TEST(AgradFwdMatrixRepVector,ffd_vector) {
  using stan::math::rep_vector;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  vector_ffd output;
  output = rep_vector(a, 4);

  EXPECT_EQ(3,output(0).val_.val());
  EXPECT_EQ(3,output(1).val_.val());
  EXPECT_EQ(3,output(2).val_.val());
  EXPECT_EQ(3,output(3).val_.val());
  EXPECT_EQ(2,output(0).d_.val());
  EXPECT_EQ(2,output(1).d_.val());
  EXPECT_EQ(2,output(2).d_.val());
  EXPECT_EQ(2,output(3).d_.val());
}

TEST(AgradFwdMatrixRepVector,ffd_vector_exception) {
  using stan::math::rep_vector;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_vector(a,-2), std::domain_error);
}
TEST(AgradFwdMatrixRepVector,ffv_vector) {
  using stan::math::rep_vector;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  fvar<fvar<var> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  vector_ffv output;
  output = rep_vector(a, 4);

  EXPECT_EQ(3,output(0).val_.val().val());
  EXPECT_EQ(3,output(1).val_.val().val());
  EXPECT_EQ(3,output(2).val_.val().val());
  EXPECT_EQ(3,output(3).val_.val().val());
  EXPECT_EQ(2,output(0).d_.val().val());
  EXPECT_EQ(2,output(1).d_.val().val());
  EXPECT_EQ(2,output(2).d_.val().val());
  EXPECT_EQ(2,output(3).d_.val().val());
}

TEST(AgradFwdMatrixRepVector,ffv_vector_exception) {
  using stan::math::rep_vector;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  fvar<fvar<var> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_vector(a,-2), std::domain_error);
}
