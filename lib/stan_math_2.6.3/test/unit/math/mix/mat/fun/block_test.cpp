#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/block.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>

TEST(AgradMixMatrixBlock,matrix_fv) {
  using stan::math::block;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(4.0,2.0);
  fvar<var> c(9.0,3.0);
  matrix_fv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  matrix_fv m = block(v, 1,1,3,3);
  EXPECT_EQ(1,m(0,0).val_.val());
  EXPECT_EQ(4,m(0,1).val_.val());
  EXPECT_EQ(9,m(0,2).val_.val());
  EXPECT_EQ(1,m(1,0).val_.val());
  EXPECT_EQ(4,m(1,1).val_.val());
  EXPECT_EQ(9,m(1,2).val_.val());
  EXPECT_EQ(1,m(2,0).val_.val());
  EXPECT_EQ(4,m(2,1).val_.val());
  EXPECT_EQ(9,m(2,2).val_.val());
  EXPECT_EQ(1,m(0,0).val_.val());
  EXPECT_EQ(2,m(0,1).d_.val());
  EXPECT_EQ(3,m(0,2).d_.val());
  EXPECT_EQ(1,m(1,0).d_.val());
  EXPECT_EQ(2,m(1,1).d_.val());
  EXPECT_EQ(3,m(1,2).d_.val());
  EXPECT_EQ(1,m(2,0).d_.val());
  EXPECT_EQ(2,m(2,1).d_.val());
  EXPECT_EQ(3,m(2,2).d_.val());

  matrix_fv n = block(v, 2,2,2,2);
  EXPECT_EQ(4,n(0,0).val_.val());
  EXPECT_EQ(9,n(0,1).val_.val());
  EXPECT_EQ(4,n(1,0).val_.val());
  EXPECT_EQ(9,n(1,1).val_.val());
  EXPECT_EQ(2,n(0,0).d_.val());
  EXPECT_EQ(3,n(0,1).d_.val());
  EXPECT_EQ(2,n(1,0).d_.val());
  EXPECT_EQ(3,n(1,1).d_.val());
}
TEST(AgradMixMatrixBlock,matrix_fv_exception) {
  using stan::math::block;
  using stan::math::matrix_fv;

  matrix_fv v(3,3);
  EXPECT_THROW(block(v,0,0,1,1), std::out_of_range);
  EXPECT_THROW(block(v,1,1,4,4), std::out_of_range);
}
TEST(AgradMixMatrixBlock,matrix_ffv) {
  using stan::math::block;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a;
  fvar<fvar<var> > b;
  fvar<fvar<var> > c;

  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 4.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 9.0;
  c.d_.val_ = 3.0;

  matrix_ffv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  matrix_ffv m = block(v, 1,1,3,3);
  EXPECT_EQ(1,m(0,0).val_.val().val());
  EXPECT_EQ(4,m(0,1).val_.val().val());
  EXPECT_EQ(9,m(0,2).val_.val().val());
  EXPECT_EQ(1,m(1,0).val_.val().val());
  EXPECT_EQ(4,m(1,1).val_.val().val());
  EXPECT_EQ(9,m(1,2).val_.val().val());
  EXPECT_EQ(1,m(2,0).val_.val().val());
  EXPECT_EQ(4,m(2,1).val_.val().val());
  EXPECT_EQ(9,m(2,2).val_.val().val());
  EXPECT_EQ(1,m(0,0).val_.val().val());
  EXPECT_EQ(2,m(0,1).d_.val().val());
  EXPECT_EQ(3,m(0,2).d_.val().val());
  EXPECT_EQ(1,m(1,0).d_.val().val());
  EXPECT_EQ(2,m(1,1).d_.val().val());
  EXPECT_EQ(3,m(1,2).d_.val().val());
  EXPECT_EQ(1,m(2,0).d_.val().val());
  EXPECT_EQ(2,m(2,1).d_.val().val());
  EXPECT_EQ(3,m(2,2).d_.val().val());

  matrix_ffv n = block(v, 2,2,2,2);
  EXPECT_EQ(4,n(0,0).val_.val().val());
  EXPECT_EQ(9,n(0,1).val_.val().val());
  EXPECT_EQ(4,n(1,0).val_.val().val());
  EXPECT_EQ(9,n(1,1).val_.val().val());
  EXPECT_EQ(2,n(0,0).d_.val().val());
  EXPECT_EQ(3,n(0,1).d_.val().val());
  EXPECT_EQ(2,n(1,0).d_.val().val());
  EXPECT_EQ(3,n(1,1).d_.val().val());
}
TEST(AgradMixMatrixBlock,matrix_ffv_exception) {
  using stan::math::block;
  using stan::math::matrix_ffv;

  matrix_ffv v(3,3);
  EXPECT_THROW(block(v,0,0,1,1), std::out_of_range);
  EXPECT_THROW(block(v,1,1,4,4), std::out_of_range);
}

