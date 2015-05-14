#include <stan/math/prim/mat/fun/size.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradMixMatrixSize,fvar_var) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::size;
  using stan::math::fvar;
  using stan::math::var;

  vector<fvar<var> > y(6);
  EXPECT_EQ(6,size(y));

  vector<Matrix<fvar<var> ,Dynamic,Dynamic> > z(7);
  EXPECT_EQ(7,size(z));

  vector<Matrix<fvar<var> ,Dynamic,1> > a(8);
  EXPECT_EQ(8,size(a));

  vector<Matrix<fvar<var> ,1,Dynamic> > b(9);
  EXPECT_EQ(9,size(b));

  vector<vector<fvar<var> > > c(10);
  EXPECT_EQ(10,size(c));

  vector<vector<fvar<var> > > ci(10);
  EXPECT_EQ(10,size(ci));

  vector<vector<Matrix<fvar<var> ,Dynamic,Dynamic> > > d(11);
  EXPECT_EQ(11,size(d));

  vector<vector<Matrix<fvar<var> ,1,Dynamic> > > e(12);
  EXPECT_EQ(12,size(e));

  vector<vector<Matrix<fvar<var> ,Dynamic,1> > > f(13);
  EXPECT_EQ(13,size(f));

  vector<vector<vector<Matrix<fvar<var> ,Dynamic,1> > > > g(14);
  EXPECT_EQ(14,size(g));
}

TEST(AgradMixMatrixSize,fvar_fvar_var) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::size;
  using stan::math::fvar;
  using stan::math::var;

  vector<fvar<fvar<var> > > y(6);
  EXPECT_EQ(6,size(y));

  vector<Matrix<fvar<fvar<var> > ,Dynamic,Dynamic> > z(7);
  EXPECT_EQ(7,size(z));

  vector<Matrix<fvar<fvar<var> > ,Dynamic,1> > a(8);
  EXPECT_EQ(8,size(a));

  vector<Matrix<fvar<fvar<var> > ,1,Dynamic> > b(9);
  EXPECT_EQ(9,size(b));

  vector<vector<fvar<fvar<var> > > > c(10);
  EXPECT_EQ(10,size(c));

  vector<vector<fvar<fvar<var> > > > ci(10);
  EXPECT_EQ(10,size(ci));

  vector<vector<Matrix<fvar<fvar<var> > ,Dynamic,Dynamic> > > d(11);
  EXPECT_EQ(11,size(d));

  vector<vector<Matrix<fvar<fvar<var> > ,1,Dynamic> > > e(12);
  EXPECT_EQ(12,size(e));

  vector<vector<Matrix<fvar<fvar<var> > ,Dynamic,1> > > f(13);
  EXPECT_EQ(13,size(f));

  vector<vector<vector<Matrix<fvar<fvar<var> > ,Dynamic,1> > > > g(14);
  EXPECT_EQ(14,size(g));
}
