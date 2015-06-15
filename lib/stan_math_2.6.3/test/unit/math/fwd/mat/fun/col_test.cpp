#include <stan/math/prim/mat/fun/col.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>

TEST(AgradFwdMatrixCol,matrix_fd) {
  using stan::math::col;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;

  matrix_fd y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  vector_fd z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_);
  EXPECT_FLOAT_EQ(4.0,z[1].val_);
  EXPECT_FLOAT_EQ(1.0,z[0].d_);
  EXPECT_FLOAT_EQ(1.0,z[1].d_);

  vector_fd w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val_);
  EXPECT_EQ(5.0,w[1].val_);
  EXPECT_EQ(1.0,w[0].d_);
  EXPECT_EQ(1.0,w[1].d_);
}
TEST(AgradFwdMatrixCol,matrix_fd_exc0) {
  using stan::math::col;
  using stan::math::matrix_fd;

  matrix_fd y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(col(y,0),std::out_of_range);
  EXPECT_THROW(col(y,7),std::out_of_range);
}
TEST(AgradFwdMatrixCol,matrix_fd_excHigh) {
  using stan::math::col;
  using stan::math::matrix_fd;

  matrix_fd y(2,3);
  y << 1, 2, 3, 4, 5, 6;
   y(0,0).d_ = 1.0;
   y(0,1).d_ = 1.0;
   y(0,2).d_ = 1.0;
   y(1,0).d_ = 1.0;
   y(1,1).d_ = 1.0;
   y(1,2).d_ = 1.0;
  EXPECT_THROW(col(y,0),std::out_of_range);
  EXPECT_THROW(col(y,5),std::out_of_range);
}
TEST(AgradFwdMatrixCol,matrix_ffd) {
  using stan::math::col;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffd y(2,3);
  y << a,b,c,d,e,f;

  vector_ffd z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val_.val());
  EXPECT_FLOAT_EQ(4.0,z[1].val_.val());
  EXPECT_FLOAT_EQ(1.0,z[0].d_.val());
  EXPECT_FLOAT_EQ(1.0,z[1].d_.val());

  vector_ffd w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val_.val());
  EXPECT_EQ(5.0,w[1].val_.val());
  EXPECT_EQ(1.0,w[0].d_.val());
  EXPECT_EQ(1.0,w[1].d_.val());
}
TEST(AgradFwdMatrixCol,matrix_ffd_exc0) {
  using stan::math::col;
  using stan::math::matrix_ffd;
  using stan::math::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffd y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::out_of_range);
  EXPECT_THROW(col(y,7),std::out_of_range);
}
TEST(AgradFwdMatrixCol,matrix_ffd_excHigh) {
  using stan::math::col;
  using stan::math::matrix_ffd;
  using stan::math::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = 6.0;
  f.d_.val_ = 1.0;

  matrix_ffd y(2,3);
  y << a,b,c,d,e,f;

  EXPECT_THROW(col(y,0),std::out_of_range);
  EXPECT_THROW(col(y,5),std::out_of_range);
}
