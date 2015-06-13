#include <stan/math/prim/mat/fun/get_base1_lhs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, failing_in_26) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  Matrix<double,Dynamic,Dynamic> y(2,3);
  EXPECT_THROW(get_base1_lhs(y,3,1,"y",2), std::exception);
  EXPECT_THROW(get_base1_lhs(y,1,4,"y",2), std::exception);
  for (int i = 1; i <= 2; ++i)
    for (int j = 1; j <= 3; ++j)
      EXPECT_FLOAT_EQ(y(i-1,j-1), get_base1_lhs(y,i,j,"y",2));
}

TEST(MathMatrix,failing_pre_20) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  Matrix<double,Dynamic,1> y(3);
  y << 1, 2, 3;
  double z = get_base1_lhs(y,1,"y",1);
  EXPECT_FLOAT_EQ(1, z);
}
TEST(MathMatrix,get_base1_lhs_vec1) {
  using stan::math::get_base1_lhs;
  std::vector<double> x(2);
  x[0] = 10.0;
  x[1] = 20.0;
  EXPECT_FLOAT_EQ(10.0,get_base1_lhs(x,1,"x[1]",0));
  EXPECT_FLOAT_EQ(20.0,get_base1_lhs(x,2,"x[1]",0));
  
  get_base1_lhs(x,2,"x[2]",0) = 5.0;
  EXPECT_FLOAT_EQ(5.0,get_base1_lhs(x,2,"x[1]",0));

  EXPECT_THROW(get_base1_lhs(x,0,"x[0]",0),
               std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,3,"x[3]",0),
               std::out_of_range);
}
TEST(MathMatrix,get_base1_lhs_vec2) {
  using stan::math::get_base1_lhs;
  using std::vector;
  size_t M = 3;
  size_t N = 4;

  vector<vector<double> > x(M,vector<double>(N,0.0));
  

  for (size_t m = 1; m <= M; ++m)
    for (size_t n = 1; n <= N; ++n)
      x[m - 1][n - 1] = (m * 10) + n;

  for (size_t m = 1; m <= M; ++m) {
    for (size_t n = 1; n <= N; ++n) {
      double expected = x[m - 1][n - 1];
      double found = get_base1_lhs(get_base1_lhs(x, m, "x[m]",1),
                               n, "x[m][n]",2);
      EXPECT_FLOAT_EQ(expected,found);
    }
  }

  get_base1_lhs(get_base1_lhs(x,1,"",-1),2,"",-1) = 112.5;
  EXPECT_FLOAT_EQ(112.5, x[0][1]);

  EXPECT_THROW(get_base1_lhs(x,0,"",-1),std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,M+1,"",-1),std::out_of_range);
  
  EXPECT_THROW(get_base1_lhs(get_base1_lhs(x,1,"",-1), 
                         12,"",-1),
               std::out_of_range);
}
TEST(MathMatrix,get_base1_lhs_matrix) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  Matrix<double,Dynamic,Dynamic> x(4,3);
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 3; ++j)
      x(i,j) = i * j;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(x(i,j),
                      get_base1_lhs(x,i+1,j+1,"x",1));
      EXPECT_FLOAT_EQ(x(i,j),
          get_base1_lhs(x,i+1,"x",1)(0,j));
      Matrix<double,1,Dynamic> xi
        = get_base1_lhs<double>(x,i+1,"x",1);
      EXPECT_FLOAT_EQ(x(i,j),xi[j]);
      EXPECT_FLOAT_EQ(x(i,j),get_base1_lhs(xi,j+1,"xi",2));
      // this is no good because can't get ref to inside val
      // could remedy by adding const versions, but don't need for Stan GM
      // double xij = get_base1_lhs<double>(get_base1_lhs<double>(x,i+1,"x",1),
      //                                j+1,"xi",2);
    }
  }
  EXPECT_THROW(get_base1_lhs(x,10,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,100,1,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,1,100,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,0,1,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,1,0,"x",1), std::out_of_range);
}
TEST(MathMatrix,get_base1_lhs_vector) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  Matrix<double,1,Dynamic> x(3);
  x << 1, 2, 3;
  
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i), get_base1_lhs(x,i+1,"x",1));
  EXPECT_THROW(get_base1_lhs(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,100,"x",1), std::out_of_range);
}
TEST(MathMatrix,get_base1_lhs_row_vector) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  Matrix<double,Dynamic,1> x(3);
  x << 1, 2, 3;
  
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i), get_base1_lhs(x,i+1,"x",1));
  EXPECT_THROW(get_base1_lhs(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1_lhs(x,100,"x",1), std::out_of_range);
}
TEST(MathMatrix,get_base1_lhs_8) {
  using stan::math::get_base1_lhs;
  using std::vector;
  double x0(42.0);
  // ~ 4m entries ~ 32MB memory + sizes
  vector<double> x1(9,x0);
  vector<vector<double> > x2(8,x1);
  vector<vector<vector<double> > > x3(7,x2);
  vector<vector<vector<vector<double> > > > x4(6,x3);
  vector<vector<vector<vector<vector<double> > > > > x5(5,x4);
  vector<vector<vector<vector<vector<vector<double> > > > > > x6(4,x5);
  vector<vector<vector<vector<vector<vector<vector<double> > > > > > > x7(3,x6);
  vector<vector<vector<vector<vector<vector<vector<vector<double> > > > > > > > x8(2,x7);

  EXPECT_EQ(x0, x8[0][0][0][0][0][0][0][0]);
  
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
                  EXPECT_FLOAT_EQ(x8[i1][i2][i3][i4][i5][i6][i7][i8],
                                  get_base1_lhs(x8,i1+1,i2+1,i3+1,i4+1,i5+1,i6+1,i7+1,i8+1,
                                            "x8",1));
}

