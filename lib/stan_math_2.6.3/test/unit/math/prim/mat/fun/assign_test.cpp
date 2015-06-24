#include <stdexcept>
#include <vector>
#include <stan/math/prim/mat/fun/assign.hpp>
#include <stan/math/prim/mat/fun/get_base1_lhs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixAssign,int) {
  using stan::math::assign;
  int a;
  int b = 5;
  assign(a,b);
  EXPECT_EQ(5,a);
  EXPECT_EQ(5,b);

  assign(a,12);
  EXPECT_EQ(a,12);
}
TEST(MathMatrixAssign,double) {
  using stan::math::assign;

  double a;
  int b = 5;
  double c = 5.0;
  assign(a,b);
  EXPECT_FLOAT_EQ(5.0,a);
  EXPECT_FLOAT_EQ(5.0,b);

  assign(a,c);
  EXPECT_FLOAT_EQ(5.0,a);
  EXPECT_FLOAT_EQ(5.0,b);
  
  assign(a,5.2);
  EXPECT_FLOAT_EQ(5.2,a);
}
TEST(MathMatrixAssign,vectorDouble) {
  using stan::math::assign;
  using std::vector;
  
  vector<double> y(3);
  y[0] = 1.2;
  y[1] = 100;
  y[2] = -5.1;

  vector<double> x(3);
  assign(x,y);
  EXPECT_EQ(3U,x.size());
  EXPECT_EQ(3U,y.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(y[i],x[i]);

  vector<double> z(2);
  EXPECT_THROW(assign(x,z), std::invalid_argument);

  vector<int> ns(3);
  ns[0] = 1;
  ns[1] = -10;
  ns[2] = 500;

  assign(x,ns);
  EXPECT_EQ(3U,x.size());
  EXPECT_EQ(3U,ns.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(ns[i], x[i]);
}



TEST(MathMatrixAssign,eigenRowVectorDoubleToDouble) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,1,Dynamic> y(3);
  y[0] = 1.2;
  y[1] = 100;
  y[2] = -5.1;

  Matrix<double,1,Dynamic> x(3);
  assign(x,y);
  EXPECT_EQ(3,x.size());
  EXPECT_EQ(3,y.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(y[i],x[i]);
}
TEST(MathMatrixAssign,eigenRowVectorIntToDouble) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,1,Dynamic> x(3);
  x[0] = 1.2;
  x[1] = 100;
  x[2] = -5.1;

  Matrix<int,1,Dynamic> ns(3);
  ns[0] = 1;
  ns[1] = -10;
  ns[2] = 500;

  assign(x,ns);
  EXPECT_EQ(3,x.size());
  EXPECT_EQ(3,ns.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(ns[i], x[i]);
}
TEST(MathMatrixAssign,eigenRowVectorShapeMismatch) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,1,Dynamic> x(3);
  x[0] = 1.2;
  x[1] = 100;
  x[2] = -5.1;

  Matrix<double,1,Dynamic> z(2);
  EXPECT_THROW(assign(x,z), std::invalid_argument);

  Matrix<double,Dynamic,1> zz(3);
  zz << 1, 2, 3;
  EXPECT_THROW(assign(x,zz),std::invalid_argument);
  
  Matrix<double,Dynamic,Dynamic> zzz(3,1);
  zzz << 1, 2, 3;
  EXPECT_THROW(assign(x,zzz),std::invalid_argument);

  Matrix<double,Dynamic,Dynamic> zzzz(1,3);
  EXPECT_THROW(assign(x,zzzz), std::invalid_argument);
}


TEST(MathMatrixAssign,eigenMatrixDoubleToDouble) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,Dynamic> y(3,2);
  y << 1.2, 100, -5.1, 12, 1000, -5100;

  Matrix<double,Dynamic,Dynamic> x(3,2);
  assign(x,y);
  EXPECT_EQ(6,x.size());
  EXPECT_EQ(6,y.size());
  EXPECT_EQ(3,x.rows());
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,x.cols());
  EXPECT_EQ(2,y.cols());
  for (size_t i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(y(i),x(i));
}
TEST(MathMatrixAssign,eigenMatrixIntToDouble) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<int,Dynamic,Dynamic> y(3,2);
  y << 1, 2, 3, 4, 5, 6;

  Matrix<double,Dynamic,Dynamic> x(3,2);
  assign(x,y);
  EXPECT_EQ(6,x.size());
  EXPECT_EQ(6,y.size());
  EXPECT_EQ(3,x.rows());
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,x.cols());
  EXPECT_EQ(2,y.cols());
  for (size_t i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(y(i),x(i));
}
TEST(MathMatrixAssign,eigenMatrixShapeMismatch) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,Dynamic> x(2,3);
  x << 1, 2, 3, 4, 5, 6;

  Matrix<double,1,Dynamic> z(6);
  z << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(assign(x,z), std::invalid_argument);
  EXPECT_THROW(assign(z,x), std::invalid_argument);

  Matrix<double,Dynamic,1> zz(6);
  zz << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(assign(x,zz),std::invalid_argument);
  EXPECT_THROW(assign(zz,x),std::invalid_argument);
  
  Matrix<double,Dynamic,Dynamic> zzz(6,1);
  zzz << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(assign(x,zzz),std::invalid_argument);
  EXPECT_THROW(assign(zzz,x),std::invalid_argument);

}

TEST(MathMatrix,block) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  using stan::math::assign;

  Matrix<double,Dynamic,Dynamic> m(2,3);
  m << 1, 2, 3, 4, 5, 6;
  
  Matrix<double,1,Dynamic> rv(3);
  rv << 10, 100, 1000;
  
  assign(get_base1_lhs(m,1,"m",1),rv);  
  EXPECT_FLOAT_EQ(10.0, m(0,0));
  EXPECT_FLOAT_EQ(100.0, m(0,1));
  EXPECT_FLOAT_EQ(1000.0, m(0,2));
}


TEST(MathMatrix,vectorVector) {
  using std::vector;
  using stan::math::assign;
  vector<vector<double> > x(3,vector<double>(2));
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 2; ++j)
      x[i][j] = (i + 1) * (j - 10);
  
  vector<vector<double> > y(3,vector<double>(2));

  assign(y,x);
  EXPECT_EQ(3U,y.size());
  for (size_t i = 0; i < 3U; ++i) {
    EXPECT_EQ(2U,y[i].size());
    for (size_t j = 0; j < 2U; ++j) {
      EXPECT_FLOAT_EQ(x[i][j],y[i][j]);
    }
  }
}


TEST(MathMatrix,vectorVectorVector) {
  using std::vector;
  using stan::math::assign;
  vector<vector<vector<double> > > 
    x(4,vector<vector<double> >(3,vector<double>(2)));
  for (size_t k = 0; k < 4; ++k)
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 2; ++j)
        x[k][i][j] = (i + 1) * (j - 10) * (20 * k + 100);
  
  vector<vector<vector<double> > > 
    y(4,vector<vector<double> >(3,vector<double>(2)));

  assign(y,x);
  EXPECT_EQ(4U,y.size());
  for (size_t k = 0; k < 4U; ++k) {
    EXPECT_EQ(3U,y[k].size());
    for (size_t i = 0; i < 3U; ++i) {
      EXPECT_EQ(2U,y[k][i].size());
      for (size_t j = 0; j < 2U; ++j) {
        EXPECT_FLOAT_EQ(x[k][i][j],y[k][i][j]);
      }
    }
  }
}

TEST(MathMatrix,vectorEigenVector) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::assign;

  vector<Matrix<double,Dynamic,1> > x(2, Matrix<double,Dynamic,1>(3));
  for (size_t i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      x[i](j) = (i + 1) * (10 * j + 2);
  vector<Matrix<double,Dynamic,1> > y(2, Matrix<double,Dynamic,1>(3));

  assign(y,x);

  EXPECT_EQ(2U,y.size());
  for (size_t i = 0; i < 2U; ++i) {
    EXPECT_EQ(3U,y[i].size());
    for (size_t j = 0; j < 3U; ++j) {
      EXPECT_FLOAT_EQ(x[i](j), y[i](j));
    }
  }
}

TEST(MathMatrix,getAssignRow) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  using stan::math::assign;

  Matrix<double,Dynamic,Dynamic> m(2,3);
  m << 1, 2, 3, 4, 5, 6;
  
  Matrix<double,1,Dynamic> rv(3);
  rv << 10, 100, 1000;
  
  assign(get_base1_lhs(m,1,"m",1),rv);  
  EXPECT_FLOAT_EQ(10.0, m(0,0));
  EXPECT_FLOAT_EQ(100.0, m(0,1));
  EXPECT_FLOAT_EQ(1000.0, m(0,2));
}

