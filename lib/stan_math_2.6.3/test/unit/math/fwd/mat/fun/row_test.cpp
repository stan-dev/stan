#include <stan/math/prim/mat/fun/row.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>

TEST(AgradFwdMatrixRow,fd_v) {
  using stan::math::row;
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;

  matrix_fd y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  row_vector_fd z = row(y,1);
  EXPECT_EQ(3,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_);
  EXPECT_FLOAT_EQ(2.0,z[1].val_);
  EXPECT_FLOAT_EQ(3.0,z[2].val_);

  row_vector_fd w = row(y,2);
  EXPECT_EQ(3,w.size());
  EXPECT_EQ(4.0,w[0].val_);
  EXPECT_EQ(5.0,w[1].val_);
  EXPECT_EQ(6.0,w[2].val_);
}
TEST(AgradFwdMatrixRow,fd_v_exc0) {
  using stan::math::row;
  using stan::math::matrix_fd;

  matrix_fd y(2,3);
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
TEST(AgradFwdMatrixRow,fd_v_excHigh) {
  using stan::math::row;
  using stan::math::matrix_fd;

  matrix_fd y(2,3);
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
TEST(AgradFwdMatrixRow,ffd_v) {
  using stan::math::row;
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;

  matrix_ffd y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  row_vector_ffd z = row(y,1);
  EXPECT_EQ(3,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_.val());
  EXPECT_FLOAT_EQ(2.0,z[1].val_.val());
  EXPECT_FLOAT_EQ(3.0,z[2].val_.val());

  row_vector_ffd w = row(y,2);
  EXPECT_EQ(3,w.size());
  EXPECT_EQ(4.0,w[0].val_.val());
  EXPECT_EQ(5.0,w[1].val_.val());
  EXPECT_EQ(6.0,w[2].val_.val());
}
TEST(AgradFwdMatrixRow,ffd_v_exc0) {
  using stan::math::row;
  using stan::math::matrix_ffd;

  matrix_ffd y(2,3);
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
TEST(AgradFwdMatrixRow,ffd_v_excHigh) {
  using stan::math::row;
  using stan::math::matrix_ffd;

  matrix_ffd y(2,3);
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
