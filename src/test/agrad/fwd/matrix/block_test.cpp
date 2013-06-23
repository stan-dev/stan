#include <gtest/gtest.h>
#include <stan/math/matrix/block.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix,block_matrix) {
  using stan::math::block;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv v(3,3);
  v << 1, 4, 9,1, 4, 9,1, 4, 9;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 2.0;
   v(0,2).d_ = 3.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 2.0;
   v(1,2).d_ = 3.0;
   v(2,0).d_ = 1.0;
   v(2,1).d_ = 2.0;
   v(2,2).d_ = 3.0;
  matrix_fv m = block(v, 1,1,3,3);
  EXPECT_EQ(1,m(0,0).val_);
  EXPECT_EQ(4,m(0,1).val_);
  EXPECT_EQ(9,m(0,2).val_);
  EXPECT_EQ(1,m(1,0).val_);
  EXPECT_EQ(4,m(1,1).val_);
  EXPECT_EQ(9,m(1,2).val_);
  EXPECT_EQ(1,m(2,0).val_);
  EXPECT_EQ(4,m(2,1).val_);
  EXPECT_EQ(9,m(2,2).val_);
  EXPECT_EQ(1,m(0,0).val_);
  EXPECT_EQ(2,m(0,1).d_);
  EXPECT_EQ(3,m(0,2).d_);
  EXPECT_EQ(1,m(1,0).d_);
  EXPECT_EQ(2,m(1,1).d_);
  EXPECT_EQ(3,m(1,2).d_);
  EXPECT_EQ(1,m(2,0).d_);
  EXPECT_EQ(2,m(2,1).d_);
  EXPECT_EQ(3,m(2,2).d_);

  matrix_fv n = block(v, 2,2,2,2);
  EXPECT_EQ(4,n(0,0).val_);
  EXPECT_EQ(9,n(0,1).val_);
  EXPECT_EQ(4,n(1,0).val_);
  EXPECT_EQ(9,n(1,1).val_);
  EXPECT_EQ(2,n(0,0).d_);
  EXPECT_EQ(3,n(0,1).d_);
  EXPECT_EQ(2,n(1,0).d_);
  EXPECT_EQ(3,n(1,1).d_);
}
TEST(AgradFwdMatrix,block_matrix_exception) {
  using stan::math::block;
  using stan::agrad::matrix_fv;

  matrix_fv v(3,3);
  EXPECT_THROW(block(v,0,0,1,1), std::domain_error);
  EXPECT_THROW(block(v,1,1,4,4), std::domain_error);
}
