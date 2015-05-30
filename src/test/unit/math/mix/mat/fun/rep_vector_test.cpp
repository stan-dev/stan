#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/rep_vector.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>

using stan::math::var;
TEST(AgradMixMatrixRepVector,fv_vector) {
  using stan::math::rep_vector;
  using stan::math::vector_fv;
  using stan::math::fvar;
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

TEST(AgradMixMatrixRepVector,fv_vector_exception) {
  using stan::math::rep_vector;
  using stan::math::vector_fv;
  using stan::math::fvar;
  fvar<var> a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_vector(a,-2), std::domain_error);
}
TEST(AgradMixMatrixRepVector,ffv_vector) {
  using stan::math::rep_vector;
  using stan::math::vector_ffv;
  using stan::math::fvar;
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

TEST(AgradMixMatrixRepVector,ffv_vector_exception) {
  using stan::math::rep_vector;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  fvar<fvar<var> > a;
  a.val_ = 3.0;
  a.d_ = 2.0;

  EXPECT_THROW(rep_vector(a,-2), std::domain_error);
}
