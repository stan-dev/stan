#include <stan/math/prim/mat/fun/get_base1.hpp>
#include <stan/math/fwd/core.hpp>
#include <gtest/gtest.h>

using stan::math::fvar;

TEST(AgradFwdMatrixGetBase1,failing_pre_20_fd) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<fvar<double> ,Dynamic,1> y(3);
  y << 1, 2, 3;
  EXPECT_FLOAT_EQ(1, get_base1(y,1,"y",1).val_);
}
TEST(AgradFwdMatrixGetBase1,get_base1_vec1_fd) {
  using stan::math::get_base1;
  std::vector<fvar<double> > x(2);
  x[0] = 10.0;
  x[1] = 20.0;
  EXPECT_FLOAT_EQ(10.0,get_base1(x,1,"x[1]",0).val_);
  EXPECT_FLOAT_EQ(20.0,get_base1(x,2,"x[1]",0).val_);
  
  // no assign in get_base1
  // get_base1(x,2,"x[2]",0) = 5.0;
  // EXPECT_FLOAT_EQ(5.0,get_base1(x,2,"x[1]",0));

  EXPECT_THROW(get_base1(x,0,"x[0]",0),
               std::out_of_range);
  EXPECT_THROW(get_base1(x,3,"x[3]",0),
               std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_vec2_fd) {
  using stan::math::get_base1;
  using std::vector;
  size_t M = 3;
  size_t N = 4;

  vector<vector<fvar<double> > > x(M,vector<fvar<double> >(N,0.0));
  

  for (size_t m = 1; m <= M; ++m)
    for (size_t n = 1; n <= N; ++n)
      x[m - 1][n - 1] = (m * 10) + n;

  for (size_t m = 1; m <= M; ++m) {
    for (size_t n = 1; n <= N; ++n) {
      fvar<double>  expected = x[m - 1][n - 1];
      fvar<double>  found = get_base1(get_base1(x, m, "x[m]",1),
                               n, "x[m][n]",2);
      EXPECT_FLOAT_EQ(expected.val_,found.val_);
    }
  }

  // no LHS usage in get_base1
  // get_base1(get_base1(x,1,"",-1),2,"",-1) = 112.5;
  // EXPECT_FLOAT_EQ(112.5, x[0][1]);

  EXPECT_THROW(get_base1(x,0,"",-1),std::out_of_range);
  EXPECT_THROW(get_base1(x,M+1,"",-1),std::out_of_range);
  
  EXPECT_THROW(get_base1(get_base1(x,1,"",-1), 
                         12,"",-1),
               std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_matrix_fd) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<fvar<double> ,Dynamic,Dynamic> x(4,3);
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 3; ++j)
      x(i,j) = i * j;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(x(i,j).val_,
                      get_base1(x,i+1,j+1,"x",1).val_);
      EXPECT_FLOAT_EQ(x(i,j).val_,
          get_base1(x,i+1,"x",1)(0,j).val_);
      Matrix<fvar<double> ,1,Dynamic> xi
        = get_base1<fvar<double> >(x,i+1,"x",1);
      EXPECT_FLOAT_EQ(x(i,j).val_,xi[j].val_);
      EXPECT_FLOAT_EQ(x(i,j).val_,get_base1(xi,j+1,"xi",2).val_);
    }
  }
  EXPECT_THROW(get_base1(x,10,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,1,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,1,100,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,0,1,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,1,0,"x",1), std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_vector_fd) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<fvar<double> ,1,Dynamic> x(3);
  x << 1, 2, 3;
  
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i).val_, get_base1(x,i+1,"x",1).val_);
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,"x",1), std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_row_vector_fd) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<fvar<double> ,Dynamic,1> x(3);
  x << 1, 2, 3;
  
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i).val_, get_base1(x,i+1,"x",1).val_);
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,"x",1), std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_8_fd) {
  using stan::math::get_base1;
  using std::vector;
  fvar<double>  x0(42.0);
  // ~ 4m entries ~ 32MB memory + sizes
  vector<fvar<double> > x1(9,x0);
  vector<vector<fvar<double> > > x2(8,x1);
  vector<vector<vector<fvar<double> > > > x3(7,x2);
  vector<vector<vector<vector<fvar<double> > > > > x4(6,x3);
  vector<vector<vector<vector<vector<fvar<double> > > > > > x5(5,x4);
  vector<vector<vector<vector<vector<vector<fvar<double> > > > > > > x6(4,x5);
  vector<vector<vector<vector<vector<vector<vector<fvar<double> > > > > > > > x7(3,x6);
  vector<vector<vector<vector<vector<vector<vector<vector<fvar<double> > > > > > > > > x8(2,x7);

  EXPECT_EQ(x0.val_, x8[0][0][0][0][0][0][0][0].val_);
  
  for (size_t i1 = 0; i1 < x8.size(); ++i1)
    for (size_t i2 = 0; i2 < x8[0].size(); ++i2)
      for (size_t i3 = 0; i3 < x8[0][0].size(); ++i3)
        for (size_t i4 = 0; i4 < x8[0][0][0].size(); ++i4)
          for (size_t i5 = 0; i5 < x8[0][0][0][0].size(); ++i5)
            for (size_t i6 = 0; i6 < x8[0][0][0][0][0].size(); ++i6)
              for (size_t i7 = 0; i7 < x8[0][0][0][0][0][0].size(); ++i7)
                for (size_t i8 = 0; i8 < x8[0][0][0][0][0][0][0].size(); ++i8)
                  x8[i1][i2][i3][i4][i5][i6][i7][i8]
                    = i1 * i2 * i3 * i4 * i5 * i6 * i7 * i8;

  for (size_t i1 = 0; i1 < x8.size(); ++i1)
    for (size_t i2 = 0; i2 < x8[0].size(); ++i2)
      for (size_t i3 = 0; i3 < x8[0][0].size(); ++i3)
        for (size_t i4 = 0; i4 < x8[0][0][0].size(); ++i4)
          for (size_t i5 = 0; i5 < x8[0][0][0][0].size(); ++i5)
            for (size_t i6 = 0; i6 < x8[0][0][0][0][0].size(); ++i6)
              for (size_t i7 = 0; i7 < x8[0][0][0][0][0][0].size(); ++i7)
                for (size_t i8 = 0; i8 < x8[0][0][0][0][0][0][0].size(); ++i8)
                  EXPECT_FLOAT_EQ(x8[i1][i2][i3][i4][i5][i6][i7][i8].val_,
                                  get_base1(x8,i1+1,i2+1,i3+1,i4+1,i5+1,i6+1,i7+1,i8+1,
                                            "x8",1).val_);
}


TEST(AgradFwdMatrixGetBase1,failing_pre_20_ffd) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<fvar<fvar<double> > ,Dynamic,1> y(3);
  y << 1, 2, 3;
  EXPECT_FLOAT_EQ(1, get_base1(y,1,"y",1).val_.val_);
}
TEST(AgradFwdMatrixGetBase1,get_base1_vec1_ffd) {
  using stan::math::get_base1;
  std::vector<fvar<fvar<double> > > x(2);
  x[0] = 10.0;
  x[1] = 20.0;
  EXPECT_FLOAT_EQ(10.0,get_base1(x,1,"x[1]",0).val_.val_);
  EXPECT_FLOAT_EQ(20.0,get_base1(x,2,"x[1]",0).val_.val_);
  
  // no assign in get_base1
  // get_base1(x,2,"x[2]",0) = 5.0;
  // EXPECT_FLOAT_EQ(5.0,get_base1(x,2,"x[1]",0));

  EXPECT_THROW(get_base1(x,0,"x[0]",0),
               std::out_of_range);
  EXPECT_THROW(get_base1(x,3,"x[3]",0),
               std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_vec2_ffd) {
  using stan::math::get_base1;
  using std::vector;
  size_t M = 3;
  size_t N = 4;

  vector<vector<fvar<fvar<double> > > > x(M,vector<fvar<fvar<double> > >(N,0.0));
  

  for (size_t m = 1; m <= M; ++m)
    for (size_t n = 1; n <= N; ++n)
      x[m - 1][n - 1] = (m * 10) + n;

  for (size_t m = 1; m <= M; ++m) {
    for (size_t n = 1; n <= N; ++n) {
      fvar<fvar<double> >  expected = x[m - 1][n - 1];
      fvar<fvar<double> >  found = get_base1(get_base1(x, m, "x[m]",1),
                               n, "x[m][n]",2);
      EXPECT_FLOAT_EQ(expected.val_.val_,found.val_.val_);
    }
  }

  // no LHS usage in get_base1
  // get_base1(get_base1(x,1,"",-1),2,"",-1) = 112.5;
  // EXPECT_FLOAT_EQ(112.5, x[0][1]);

  EXPECT_THROW(get_base1(x,0,"",-1),std::out_of_range);
  EXPECT_THROW(get_base1(x,M+1,"",-1),std::out_of_range);
  
  EXPECT_THROW(get_base1(get_base1(x,1,"",-1), 
                         12,"",-1),
               std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_matrix_ffd) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<fvar<fvar<double> > ,Dynamic,Dynamic> x(4,3);
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 3; ++j)
      x(i,j) = i * j;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(x(i,j).val_.val_,
                      get_base1(x,i+1,j+1,"x",1).val_.val_);
      EXPECT_FLOAT_EQ(x(i,j).val_.val_,
          get_base1(x,i+1,"x",1)(0,j).val_.val_);
      Matrix<fvar<fvar<double> > ,1,Dynamic> xi
        = get_base1<fvar<fvar<double> > >(x,i+1,"x",1);
      EXPECT_FLOAT_EQ(x(i,j).val_.val_,xi[j].val_.val_);
      EXPECT_FLOAT_EQ(x(i,j).val_.val_,get_base1(xi,j+1,"xi",2).val_.val_);
    }
  }
  EXPECT_THROW(get_base1(x,10,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,1,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,1,100,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,0,1,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,1,0,"x",1), std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_vector_ffd) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<fvar<fvar<double> > ,1,Dynamic> x(3);
  x << 1, 2, 3;
  
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i).val_.val_, get_base1(x,i+1,"x",1).val_.val_);
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,"x",1), std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_row_vector_ffd) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<fvar<fvar<double> > ,Dynamic,1> x(3);
  x << 1, 2, 3;
  
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i).val_.val_, get_base1(x,i+1,"x",1).val_.val_);
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,"x",1), std::out_of_range);
}
TEST(AgradFwdMatrixGetBase1,get_base1_8_ffd) {
  using stan::math::get_base1;
  using std::vector;
  fvar<fvar<double> >  x0(42.0);
  // ~ 4m entries ~ 32MB memory + sizes
  vector<fvar<fvar<double> > > x1(9,x0);
  vector<vector<fvar<fvar<double> > > > x2(8,x1);
  vector<vector<vector<fvar<fvar<double> > > > > x3(7,x2);
  vector<vector<vector<vector<fvar<fvar<double> > > > > > x4(6,x3);
  vector<vector<vector<vector<vector<fvar<fvar<double> > > > > > > x5(5,x4);
  vector<vector<vector<vector<vector<vector<fvar<fvar<double> > > > > > > > x6(4,x5);
  vector<vector<vector<vector<vector<vector<vector<fvar<fvar<double> > > > > > > > > x7(3,x6);
  vector<vector<vector<vector<vector<vector<vector<vector<fvar<fvar<double> > > > > > > > > > x8(2,x7);

  EXPECT_EQ(x0.val_.val_, x8[0][0][0][0][0][0][0][0].val_.val_);
  
  for (size_t i1 = 0; i1 < x8.size(); ++i1)
    for (size_t i2 = 0; i2 < x8[0].size(); ++i2)
      for (size_t i3 = 0; i3 < x8[0][0].size(); ++i3)
        for (size_t i4 = 0; i4 < x8[0][0][0].size(); ++i4)
          for (size_t i5 = 0; i5 < x8[0][0][0][0].size(); ++i5)
            for (size_t i6 = 0; i6 < x8[0][0][0][0][0].size(); ++i6)
              for (size_t i7 = 0; i7 < x8[0][0][0][0][0][0].size(); ++i7)
                for (size_t i8 = 0; i8 < x8[0][0][0][0][0][0][0].size(); ++i8)
                  x8[i1][i2][i3][i4][i5][i6][i7][i8]
                    = i1 * i2 * i3 * i4 * i5 * i6 * i7 * i8;

  for (size_t i1 = 0; i1 < x8.size(); ++i1)
    for (size_t i2 = 0; i2 < x8[0].size(); ++i2)
      for (size_t i3 = 0; i3 < x8[0][0].size(); ++i3)
        for (size_t i4 = 0; i4 < x8[0][0][0].size(); ++i4)
          for (size_t i5 = 0; i5 < x8[0][0][0][0].size(); ++i5)
            for (size_t i6 = 0; i6 < x8[0][0][0][0][0].size(); ++i6)
              for (size_t i7 = 0; i7 < x8[0][0][0][0][0][0].size(); ++i7)
                for (size_t i8 = 0; i8 < x8[0][0][0][0][0][0][0].size(); ++i8)
                  EXPECT_FLOAT_EQ(x8[i1][i2][i3][i4][i5][i6][i7][i8].val_.val_,
                                  get_base1(x8,i1+1,i2+1,i3+1,i4+1,i5+1,i6+1,i7+1,i8+1,
                                            "x8",1).val_.val_);
}
