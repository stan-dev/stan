#include <gtest/gtest.h>
#include <stan/math/rep_row_vector.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/var.hpp>

using stan::agrad::var;
TEST(AgradFwdMatrix,rep_rowvector) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  row_vector_fv output;
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

TEST(AgradFwdMatrix,rep_rowvector_exception) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
TEST(AgradFvarVarMatrix,rep_rowvector) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_fvv;
  using stan::agrad::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  row_vector_fvv output;
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

TEST(AgradFvarVarMatrix,rep_rowvector_exception) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_fvv;
  using stan::agrad::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
TEST(AgradFvarFvarMatrix,rep_rowvector) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  row_vector_ffv output;
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

TEST(AgradFvarFvarMatrix,rep_rowvector_exception) {
  using stan::math::rep_row_vector;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  fvar<fvar<double> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_row_vector(a,-2), std::domain_error);
}
