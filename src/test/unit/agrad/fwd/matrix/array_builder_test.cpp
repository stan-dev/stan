#include <stan/math/matrix/array_builder.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>

using stan::agrad::fvar;
using stan::agrad::var;

TEST(AgradFwdMatrixArrayBuilder,fvar_double) {
  using std::vector;
  using stan::math::array_builder;

  EXPECT_EQ(0U, array_builder<fvar<double> >().array().size());

  vector<fvar<double> > x
    = array_builder<fvar<double> >()
    .add(fvar<double>(1,4))
    .add(fvar<double>(3,4))
    .add(fvar<double>(2,4))
    .array();
  EXPECT_EQ(3U,x.size());
  EXPECT_FLOAT_EQ(1.0, x[0].val_);
  EXPECT_FLOAT_EQ(3.0, x[1].val_);
  EXPECT_FLOAT_EQ(2.0, x[2].val_);
  EXPECT_FLOAT_EQ(4.0, x[0].d_);
  EXPECT_FLOAT_EQ(4.0, x[1].d_);
  EXPECT_FLOAT_EQ(4.0, x[2].d_);

  vector<vector<fvar<double> > > xx
    = array_builder<vector<fvar<double> > >()
    .add(array_builder<fvar<double> >()
         .add(fvar<double>(1,4))
         .add(fvar<double>(2,4)).array())
    .add(array_builder<fvar<double> >()
         .add(fvar<double>(3,4))
         .add(fvar<double>(4,4)).array())
    .add(array_builder<fvar<double> >()
         .add(fvar<double>(5,4))
         .add(fvar<double>(6,4)).array())
    .array();

  EXPECT_EQ(3U,xx.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(2U,xx[i].size());
  EXPECT_EQ(1,xx[0][0].val_);
  EXPECT_EQ(2,xx[0][1].val_);
  EXPECT_EQ(3,xx[1][0].val_);
  EXPECT_EQ(4,xx[1][1].val_);
  EXPECT_EQ(5,xx[2][0].val_);
  EXPECT_EQ(6,xx[2][1].val_);
  EXPECT_EQ(4,xx[0][0].d_);
  EXPECT_EQ(4,xx[0][1].d_);
  EXPECT_EQ(4,xx[1][0].d_);
  EXPECT_EQ(4,xx[1][1].d_);
  EXPECT_EQ(4,xx[2][0].d_);
  EXPECT_EQ(4,xx[2][1].d_);
}


TEST(AgradFwdMatrixArrayBuilder,fvar_fvar_double) {
  using std::vector;
  using stan::math::array_builder;

  EXPECT_EQ(0U, array_builder<fvar<fvar<double> > >().array().size());

  vector<fvar<fvar<double> > > x
    = array_builder<fvar<fvar<double> > >()
    .add(fvar<fvar<double> >(1,4))
    .add(fvar<fvar<double> >(3,4))
    .add(fvar<fvar<double> >(2,4))
    .array();
  EXPECT_EQ(3U,x.size());
  EXPECT_FLOAT_EQ(1.0, x[0].val_.val_);
  EXPECT_FLOAT_EQ(3.0, x[1].val_.val_);
  EXPECT_FLOAT_EQ(2.0, x[2].val_.val_);
  EXPECT_FLOAT_EQ(4.0, x[0].d_.val_);
  EXPECT_FLOAT_EQ(4.0, x[1].d_.val_);
  EXPECT_FLOAT_EQ(4.0, x[2].d_.val_);

  vector<vector<fvar<fvar<double> > > > xx
    = array_builder<vector<fvar<fvar<double> > > >()
    .add(array_builder<fvar<fvar<double> > >()
         .add(fvar<fvar<double> >(1,4))
         .add(fvar<fvar<double> >(2,4)).array())
    .add(array_builder<fvar<fvar<double> > >()
         .add(fvar<fvar<double> >(3,4))
         .add(fvar<fvar<double> >(4,4)).array())
    .add(array_builder<fvar<fvar<double> > >()
         .add(fvar<fvar<double> >(5,4))
         .add(fvar<fvar<double> >(6,4)).array())
    .array();

  EXPECT_EQ(3U,xx.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(2U,xx[i].size());
  EXPECT_EQ(1,xx[0][0].val_.val_);
  EXPECT_EQ(2,xx[0][1].val_.val_);
  EXPECT_EQ(3,xx[1][0].val_.val_);
  EXPECT_EQ(4,xx[1][1].val_.val_);
  EXPECT_EQ(5,xx[2][0].val_.val_);
  EXPECT_EQ(6,xx[2][1].val_.val_);
  EXPECT_EQ(4,xx[0][0].d_.val_);
  EXPECT_EQ(4,xx[0][1].d_.val_);
  EXPECT_EQ(4,xx[1][0].d_.val_);
  EXPECT_EQ(4,xx[1][1].d_.val_);
  EXPECT_EQ(4,xx[2][0].d_.val_);
  EXPECT_EQ(4,xx[2][1].d_.val_);
}


TEST(AgradFwdMatrixArrayBuilder,fvar_var) {
  using std::vector;
  using stan::math::array_builder;

  EXPECT_EQ(0U, array_builder<fvar<var> >().array().size());

  vector<fvar<var> > x
    = array_builder<fvar<var> >()
    .add(fvar<var>(1,4))
    .add(fvar<var>(3,4))
    .add(fvar<var>(2,4))
    .array();
  EXPECT_EQ(3U,x.size());
  EXPECT_FLOAT_EQ(1.0, x[0].val_.val());
  EXPECT_FLOAT_EQ(3.0, x[1].val_.val());
  EXPECT_FLOAT_EQ(2.0, x[2].val_.val());
  EXPECT_FLOAT_EQ(4.0, x[0].d_.val());
  EXPECT_FLOAT_EQ(4.0, x[1].d_.val());
  EXPECT_FLOAT_EQ(4.0, x[2].d_.val());

  vector<vector<fvar<var> > > xx
    = array_builder<vector<fvar<var> > >()
    .add(array_builder<fvar<var> >()
         .add(fvar<var>(1,4))
         .add(fvar<var>(2,4)).array())
    .add(array_builder<fvar<var> >()
         .add(fvar<var>(3,4))
         .add(fvar<var>(4,4)).array())
    .add(array_builder<fvar<var> >()
         .add(fvar<var>(5,4))
         .add(fvar<var>(6,4)).array())
    .array();

  EXPECT_EQ(3U,xx.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(2U,xx[i].size());
  EXPECT_EQ(1,xx[0][0].val_.val());
  EXPECT_EQ(2,xx[0][1].val_.val());
  EXPECT_EQ(3,xx[1][0].val_.val());
  EXPECT_EQ(4,xx[1][1].val_.val());
  EXPECT_EQ(5,xx[2][0].val_.val());
  EXPECT_EQ(6,xx[2][1].val_.val());
  EXPECT_EQ(4,xx[0][0].d_.val());
  EXPECT_EQ(4,xx[0][1].d_.val());
  EXPECT_EQ(4,xx[1][0].d_.val());
  EXPECT_EQ(4,xx[1][1].d_.val());
  EXPECT_EQ(4,xx[2][0].d_.val());
  EXPECT_EQ(4,xx[2][1].d_.val());
}


TEST(AgradFwdMatrixArrayBuilder,fvar_fvar_var) {
  using std::vector;
  using stan::math::array_builder;

  EXPECT_EQ(0U, array_builder<fvar<fvar<var> > >().array().size());

  vector<fvar<fvar<var> > > x
    = array_builder<fvar<fvar<var> > >()
    .add(fvar<fvar<var> >(1,4))
    .add(fvar<fvar<var> >(3,4))
    .add(fvar<fvar<var> >(2,4))
    .array();
  EXPECT_EQ(3U,x.size());
  EXPECT_FLOAT_EQ(1.0, x[0].val_.val_.val());
  EXPECT_FLOAT_EQ(3.0, x[1].val_.val_.val());
  EXPECT_FLOAT_EQ(2.0, x[2].val_.val_.val());
  EXPECT_FLOAT_EQ(4.0, x[0].d_.val_.val());
  EXPECT_FLOAT_EQ(4.0, x[1].d_.val_.val());
  EXPECT_FLOAT_EQ(4.0, x[2].d_.val_.val());

  vector<vector<fvar<fvar<var> > > > xx
    = array_builder<vector<fvar<fvar<var> > > >()
    .add(array_builder<fvar<fvar<var> > >()
         .add(fvar<fvar<var> >(1,4))
         .add(fvar<fvar<var> >(2,4)).array())
    .add(array_builder<fvar<fvar<var> > >()
         .add(fvar<fvar<var> >(3,4))
         .add(fvar<fvar<var> >(4,4)).array())
    .add(array_builder<fvar<fvar<var> > >()
         .add(fvar<fvar<var> >(5,4))
         .add(fvar<fvar<var> >(6,4)).array())
    .array();

  EXPECT_EQ(3U,xx.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(2U,xx[i].size());
  EXPECT_EQ(1,xx[0][0].val_.val_.val());
  EXPECT_EQ(2,xx[0][1].val_.val_.val());
  EXPECT_EQ(3,xx[1][0].val_.val_.val());
  EXPECT_EQ(4,xx[1][1].val_.val_.val());
  EXPECT_EQ(5,xx[2][0].val_.val_.val());
  EXPECT_EQ(6,xx[2][1].val_.val_.val());
  EXPECT_EQ(4,xx[0][0].d_.val_.val());
  EXPECT_EQ(4,xx[0][1].d_.val_.val());
  EXPECT_EQ(4,xx[1][0].d_.val_.val());
  EXPECT_EQ(4,xx[1][1].d_.val_.val());
  EXPECT_EQ(4,xx[2][0].d_.val_.val());
  EXPECT_EQ(4,xx[2][1].d_.val_.val());
}

