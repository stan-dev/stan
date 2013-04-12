#include <stan/math/matrix/col.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix,col_v) {
  using stan::math::col;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  vector_fv z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_);
  EXPECT_FLOAT_EQ(4.0,z[1].val_);
  EXPECT_FLOAT_EQ(1.0,z[0].d_);
  EXPECT_FLOAT_EQ(1.0,z[1].d_);

  vector_fv w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val_);
  EXPECT_EQ(5.0,w[1].val_);
  EXPECT_EQ(1.0,w[0].d_);
  EXPECT_EQ(1.0,w[1].d_);
}
TEST(AgradFwdMatrix,col_v_exc0) {
  using stan::math::col;
  using stan::agrad::matrix_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,7),std::domain_error);
}
TEST(AgradFwdMatrix,col_v_excHigh) {
  using stan::math::col;
  using stan::agrad::matrix_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,5),std::domain_error);
}
