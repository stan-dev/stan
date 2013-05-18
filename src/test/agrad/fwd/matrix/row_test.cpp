#include <stan/math/matrix/row.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix,row_v) {
  using stan::math::row;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  row_vector_fv z = row(y,1);
  EXPECT_EQ(3,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_);
  EXPECT_FLOAT_EQ(2.0,z[1].val_);
  EXPECT_FLOAT_EQ(3.0,z[2].val_);

  row_vector_fv w = row(y,2);
  EXPECT_EQ(3,w.size());
  EXPECT_EQ(4.0,w[0].val_);
  EXPECT_EQ(5.0,w[1].val_);
  EXPECT_EQ(6.0,w[2].val_);
}
TEST(AgradFwdMatrix,row_v_exc0) {
  using stan::math::row;
  using stan::agrad::matrix_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(row(y,0),std::domain_error);
  EXPECT_THROW(row(y,7),std::domain_error);
}
TEST(AgradFwdMatrix,row_v_excHigh) {
  using stan::math::row;
  using stan::agrad::matrix_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(row(y,0),std::domain_error);
  EXPECT_THROW(row(y,5),std::domain_error);
}
