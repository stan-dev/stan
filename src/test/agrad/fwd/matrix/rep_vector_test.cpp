#include <gtest/gtest.h>
#include <stan/math/rep_vector.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>

TEST(AgradFwdMatrix,rep_vector) {
  using stan::math::rep_vector;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;
  vector_fv output;
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

TEST(AgradFwdMatrix,rep_vector_exception) {
  using stan::math::rep_vector;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  fvar<double> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_vector(a,-2), std::domain_error);
}
