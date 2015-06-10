#include <stan/math/prim/mat/fun/row.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>

TEST(AgradMixMatrixRow,fv_v) {
  using stan::math::row;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;

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
  EXPECT_FLOAT_EQ(1.0,z[0].val_.val());
  EXPECT_FLOAT_EQ(2.0,z[1].val_.val());
  EXPECT_FLOAT_EQ(3.0,z[2].val_.val());

  row_vector_fv w = row(y,2);
  EXPECT_EQ(3,w.size());
  EXPECT_EQ(4.0,w[0].val_.val());
  EXPECT_EQ(5.0,w[1].val_.val());
  EXPECT_EQ(6.0,w[2].val_.val());
}
TEST(AgradMixMatrixRow,fv_v_exc0) {
  using stan::math::row;
  using stan::math::matrix_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(row(y,0),std::out_of_range);
  EXPECT_THROW(row(y,7),std::out_of_range);
}
TEST(AgradMixMatrixRow,fv_v_excHigh) {
  using stan::math::row;
  using stan::math::matrix_fv;

  matrix_fv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(row(y,0),std::out_of_range);
  EXPECT_THROW(row(y,5),std::out_of_range);
}
TEST(AgradMixMatrixRow,ffv_v) {
  using stan::math::row;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_ffv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  row_vector_ffv z = row(y,1);
  EXPECT_EQ(3,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_.val().val());
  EXPECT_FLOAT_EQ(2.0,z[1].val_.val().val());
  EXPECT_FLOAT_EQ(3.0,z[2].val_.val().val());

  row_vector_ffv w = row(y,2);
  EXPECT_EQ(3,w.size());
  EXPECT_EQ(4.0,w[0].val_.val().val());
  EXPECT_EQ(5.0,w[1].val_.val().val());
  EXPECT_EQ(6.0,w[2].val_.val().val());
}
TEST(AgradMixMatrixRow,ffv_v_exc0) {
  using stan::math::row;
  using stan::math::matrix_ffv;

  matrix_ffv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(row(y,0),std::out_of_range);
  EXPECT_THROW(row(y,7),std::out_of_range);
}
TEST(AgradMixMatrixRow,ffv_v_excHigh) {
  using stan::math::row;
  using stan::math::matrix_ffv;

  matrix_ffv y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(row(y,0),std::out_of_range);
  EXPECT_THROW(row(y,5),std::out_of_range);
}
