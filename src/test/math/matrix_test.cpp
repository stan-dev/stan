#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>
#include <stan/math/matrix.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

using stan::math::matrix_d;
using stan::math::vector_d;
using stan::math::row_vector_d;

// typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_d;
// typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_d;
// typedef Eigen::Matrix<double,1,Eigen::Dynamic> row_vector_d;

TEST(matrixTest,col) {
  matrix_d m(3,4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  vector_d c = m.col(0);
  vector_d c2 = stan::math::col(m,1);
  EXPECT_EQ(3,c.size());
  EXPECT_EQ(3,c2.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(c[i],c2[i]);
}
TEST(matrixTest,row) {
  matrix_d m(3,4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  row_vector_d c = m.row(1);
  row_vector_d c2 = stan::math::row(m,2);
  EXPECT_EQ(4,c.size());
  EXPECT_EQ(4,c2.size());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(c[i],c2[i]);
}

TEST(matrix_test, resize_double) {
  double x = 5;
  std::vector<size_t> dims;
  stan::math::resize(x,dims);
}
TEST(matrix_test, resize_svec_double) {
  std::vector<double> y;
  std::vector<size_t> dims;
  EXPECT_EQ(0U, y.size());

  dims.push_back(4U);
  stan::math::resize(y,dims);
  EXPECT_EQ(4U, y.size());

  dims[0] = 2U;
  stan::math::resize(y,dims);
  EXPECT_EQ(2U, y.size());
}
TEST(matrix_test, resize_vec_double) {
  Matrix<double,Dynamic,1> v(2);
  std::vector<size_t> dims;
  EXPECT_EQ(2, v.size());

  dims.push_back(17U);
  stan::math::resize(v,dims);
  EXPECT_EQ(17, v.size());

  dims[0] = 3U;
  stan::math::resize(v,dims);
  EXPECT_EQ(3, v.size());
}
TEST(matrix_test, resize_rvec_double) {
  Matrix<double,1,Dynamic> rv(2);
  std::vector<size_t> dims;
  EXPECT_EQ(2, rv.size());

  dims.push_back(17U);
  stan::math::resize(rv,dims);
  EXPECT_EQ(17, rv.size());

  dims[0] = 3U;
  stan::math::resize(rv,dims);
  EXPECT_EQ(3, rv.size());
}
TEST(matrix_test, resize_mat_double) {
  Matrix<double,Dynamic,Dynamic> m(2,3);
  std::vector<size_t> dims;
  EXPECT_EQ(2, m.rows());
  EXPECT_EQ(3, m.cols());

  dims.push_back(7U);
  dims.push_back(17U);
  stan::math::resize(m,dims);
  EXPECT_EQ(7, m.rows());
  EXPECT_EQ(17, m.cols());
}
TEST(matrix_test, resize_svec_svec_double) {
  std::vector<std::vector<double> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<size_t> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::math::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5U,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::math::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7U,xx[1].size());  
}
TEST(matrix_test, resize_svec_v_double) {
  std::vector<Matrix<double,Dynamic,1> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<size_t> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::math::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::math::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7,xx[1].size());  
}
TEST(matrix_test, resize_svec_rv_double) {
  std::vector<Matrix<double,1,Dynamic> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<size_t> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::math::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::math::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7,xx[1].size());  
}
TEST(matrix_test, resize_svec_svec_matrix_double) {
  std::vector<std::vector<Matrix<double,Dynamic,Dynamic> > > mm;
  std::vector<size_t> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  dims.push_back(6U);
  dims.push_back(3U);
  stan::math::resize(mm,dims);
  EXPECT_EQ(4U,mm.size());
  EXPECT_EQ(5U,mm[0].size());
  EXPECT_EQ(6,mm[1][2].rows());
  EXPECT_EQ(3,mm[3][4].cols());
}

TEST(matrix,get_base1_vec1) {
  using stan::math::get_base1;
  std::vector<double> x(2);
  x[0] = 10.0;
  x[1] = 20.0;
  EXPECT_FLOAT_EQ(10.0,get_base1(x,1,"x[1]",0));
  EXPECT_FLOAT_EQ(20.0,get_base1(x,2,"x[1]",0));
  
  get_base1(x,2,"x[2]",0) = 5.0;
  EXPECT_FLOAT_EQ(5.0,get_base1(x,2,"x[1]",0));

  EXPECT_THROW(get_base1(x,0,"x[0]",0),
               std::out_of_range);
  EXPECT_THROW(get_base1(x,3,"x[3]",0),
               std::out_of_range);
}
TEST(matrix,get_base1_vec2) {
  using stan::math::get_base1;
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
      double found = get_base1(get_base1(x, m, "x[m]",1),
                               n, "x[m][n]",2);
      EXPECT_FLOAT_EQ(expected,found);
    }
  }

  get_base1(get_base1(x,1,"",-1),2,"",-1) = 112.5;
  EXPECT_FLOAT_EQ(112.5, x[0][1]);

  EXPECT_THROW(get_base1(x,0,"",-1),std::out_of_range);
  EXPECT_THROW(get_base1(x,M+1,"",-1),std::out_of_range);
  
  EXPECT_THROW(get_base1(get_base1(x,1,"",-1), 
                         12,"",-1),
               std::out_of_range);
}
TEST(matrix,get_base1_matrix) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<double,Dynamic,Dynamic> x(4,3);
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 3; ++j)
      x(i,j) = i * j;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(x(i,j),
                      get_base1(x,i+1,j+1,"x",1));
      EXPECT_FLOAT_EQ(x(i,j),
                      get_base1(x,i+1,"x",1)[j]);
      Matrix<double,Dynamic,1> xi
        = get_base1<double>(x,i+1,"x",1);
      EXPECT_FLOAT_EQ(x(i,j),xi[j]);
      EXPECT_FLOAT_EQ(x(i,j),get_base1(xi,j+1,"xi",2));
      // this is no good because can't get ref to inside val
      // could remedy by adding const versions, but don't need for Stan GM
      // double xij = get_base1<double>(get_base1<double>(x,i+1,"x",1),
      //                                j+1,"xi",2);
    }
  }
  EXPECT_THROW(get_base1(x,10,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,1,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,1,100,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,0,1,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,1,0,"x",1), std::out_of_range);
}
TEST(matrix,get_base1_vector) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<double,1,Dynamic> x(3);
  x << 1, 2, 3;
  
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i), get_base1(x,i+1,"x",1));
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,"x",1), std::out_of_range);
}
TEST(matrix,get_base1_row_vector) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1;
  Matrix<double,Dynamic,1> x(3);
  x << 1, 2, 3;
  
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i), get_base1(x,i+1,"x",1));
  EXPECT_THROW(get_base1(x,0,"x",1), std::out_of_range);
  EXPECT_THROW(get_base1(x,100,"x",1), std::out_of_range);
}
TEST(matrix,get_base1_8) {
  using stan::math::get_base1;
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
                                  get_base1(x8,i1+1,i2+1,i3+1,i4+1,i5+1,i6+1,i7+1,i8+1,
                                            "x8",1));
}

// exp tests
TEST(matrix_test, exp__matrix) {
  matrix_d expected_output(2,2);
  matrix_d mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = stan::math::exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j));
}

// log tests
TEST(matrix_test, log__matrix) {
  matrix_d expected_output(2,2);
  matrix_d mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = stan::math::log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j));
}

TEST(matrix_test,add_v_exception) {
  vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::math::add(d1, d2), std::domain_error);
}
TEST(matrix_test,add_rv_exception) {
  row_vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::math::add(d1, d2), std::domain_error);
}
TEST(matrix_test,add_m_exception) {
  matrix_d d1, d2;

  d1.resize(2,3);
  d2.resize(2,3);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(0,0);
  d2.resize(0,0);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(2,3);
  d2.resize(3,3);
  EXPECT_THROW(stan::math::add(d1, d2), std::domain_error);
}

TEST(matrix_test,subtract_v_exception) {
  vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::math::subtract(d1, d2), std::domain_error);
}
TEST(matrix_test,subtract_rv_exception) {
  row_vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::math::subtract(d1, d2), std::domain_error);
}
TEST(matrix_test,subtract_m_exception) {
  matrix_d d1, d2;

  d1.resize(2,3);
  d2.resize(2,3);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(0,0);
  d2.resize(0,0);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(2,3);
  d2.resize(3,3);
  EXPECT_THROW(stan::math::subtract(d1, d2), std::domain_error);
}

TEST(matrixTest,subtract_c_m) {
  matrix_d v(2,2);
  v << 1, 2, 3, 4;
  matrix_d result;

  result = stan::math::subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0,0));
  EXPECT_FLOAT_EQ(0.0,result(0,1));
  EXPECT_FLOAT_EQ(-1.0,result(1,0));
  EXPECT_FLOAT_EQ(-2.0,result(1,1));

  result = stan::math::subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0,0));
  EXPECT_FLOAT_EQ(0.0,result(0,1));
  EXPECT_FLOAT_EQ(1.0,result(1,0));
  EXPECT_FLOAT_EQ(2.0,result(1,1));
}

TEST(matrixTest,subtract_c_rv) {
  row_vector_d v(3);
  v << 1, 2, 3;
  row_vector_d result;

  result = stan::math::subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0));
  EXPECT_FLOAT_EQ(0.0,result(1));
  EXPECT_FLOAT_EQ(-1.0,result(2));

  result = stan::math::subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0));
  EXPECT_FLOAT_EQ(0.0,result(1));
  EXPECT_FLOAT_EQ(1.0,result(2));
}


TEST(matrixTest,subtract_c_v) {
  vector_d v(3);
  v << 1, 2, 3;
  vector_d result;

  result = stan::math::subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0));
  EXPECT_FLOAT_EQ(0.0,result(1));
  EXPECT_FLOAT_EQ(-1.0,result(2));

  result = stan::math::subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0));
  EXPECT_FLOAT_EQ(0.0,result(1));
  EXPECT_FLOAT_EQ(1.0,result(2));
}

TEST(matrixTest,add_c_m) {
  matrix_d v(2,2);
  v << 1, 2, 3, 4;
  matrix_d result;

  result = stan::math::add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0,0));
  EXPECT_FLOAT_EQ(4.0,result(0,1));
  EXPECT_FLOAT_EQ(5.0,result(1,0));
  EXPECT_FLOAT_EQ(6.0,result(1,1));

  result = stan::math::add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0,0));
  EXPECT_FLOAT_EQ(4.0,result(0,1));
  EXPECT_FLOAT_EQ(5.0,result(1,0));
  EXPECT_FLOAT_EQ(6.0,result(1,1));
}

TEST(matrixTest,add_c_rv) {
  row_vector_d v(3);
  v << 1, 2, 3;
  row_vector_d result;

  result = stan::math::add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(5.0,result(2));

  result = stan::math::add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(5.0,result(2));
}


TEST(matrixTest,add_c_v) {
  vector_d v(3);
  v << 1, 2, 3;
  vector_d result;

  result = stan::math::add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(5.0,result(2));

  result = stan::math::add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(5.0,result(2));
}

TEST(matrixTest,multiply_c_v) {
  vector_d v(3);
  v << 1, 2, 3;
  vector_d result = stan::math::multiply(2.0,v);
  EXPECT_FLOAT_EQ(2.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(6.0,result(2));
}
TEST(matrixTest,multiply_c_rv) {
  row_vector_d rv(3);
  rv << 1, 2, 3;
  row_vector_d result = stan::math::multiply(2.0,rv);
  EXPECT_FLOAT_EQ(2.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(6.0,result(2));
}
TEST(matrixTest,multiply_c_m) {
  matrix_d m(2,3);
  m << 1, 2, 3, 4, 5, 6;
  matrix_d result = stan::math::multiply(2.0,m);
  EXPECT_FLOAT_EQ(2.0,result(0,0));
  EXPECT_FLOAT_EQ(4.0,result(0,1));
  EXPECT_FLOAT_EQ(6.0,result(0,2));
  EXPECT_FLOAT_EQ(8.0,result(1,0));
  EXPECT_FLOAT_EQ(10.0,result(1,1));
  EXPECT_FLOAT_EQ(12.0,result(1,2));
}

TEST(matrix_test,multiply_rv_v_exception) {
  row_vector_d rv;
  vector_d v;
  
  rv.resize(3);
  v.resize(3);
  EXPECT_NO_THROW(stan::math::multiply(rv, v));

  rv.resize(0);
  v.resize(0);
  EXPECT_NO_THROW(stan::math::multiply(rv, v));

  rv.resize(2);
  v.resize(3);
  EXPECT_THROW(stan::math::multiply(rv, v), std::domain_error);
}
TEST(matrix_test,multiply_m_v_exception) {
  matrix_d m;
  vector_d v;
  
  m.resize(3, 5);
  v.resize(5);
  EXPECT_NO_THROW(stan::math::multiply(m, v));

  m.resize(3, 0);
  v.resize(0);
  EXPECT_NO_THROW(stan::math::multiply(m, v));

  m.resize(2, 3);
  v.resize(2);
  EXPECT_THROW(stan::math::multiply(m, v), std::domain_error);  
}
TEST(matrix_test,multiply_rv_m_exception) {
  row_vector_d rv;
  matrix_d m;
    
  rv.resize(3);
  m.resize(3, 5);
  EXPECT_NO_THROW(stan::math::multiply(rv, m));

  rv.resize(0);
  m.resize(0, 3);
  EXPECT_NO_THROW(stan::math::multiply(rv, m));

  rv.resize(3);
  m.resize(2, 3);
  EXPECT_THROW(stan::math::multiply(rv, m), std::domain_error);
}
TEST(matrix_test,multiply_m_m_exception) {
  matrix_d m1, m2;
  
  m1.resize(1, 3);
  m2.resize(3, 5);
  EXPECT_NO_THROW(stan::math::multiply(m1, m2));

  
  m1.resize(2, 0);
  m2.resize(0, 3);
  EXPECT_NO_THROW(stan::math::multiply(m1, m2));

  m1.resize(4, 3);
  m2.resize(2, 3);
  EXPECT_THROW(stan::math::multiply(m1, m2), std::domain_error);
}
TEST(matrix_test,cholesky_decompose_exception) {
  matrix_d m;
  
  m.resize(2,2);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);
}

TEST(matrix_test,std_vector_sum_int) {
  std::vector<int> x(3);
  EXPECT_EQ(0,stan::math::sum(x));
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  EXPECT_EQ(6,stan::math::sum(x));
}
TEST(matrix_test,std_vector_sum_double) {
  using stan::math::sum;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  std::vector<double> x(3);
  EXPECT_FLOAT_EQ(0.0,sum(x));
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  EXPECT_FLOAT_EQ(6.0,sum(x));

  vector_d v;
  EXPECT_FLOAT_EQ(0.0,sum(v));
  v = vector_d(1);
  v[0] = 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(v));
  v = vector_d(3);
  v[0] = 5.0;
  v[1] = 10.0;
  v[2] = 100.0;
  EXPECT_FLOAT_EQ(115.0,sum(v));

  row_vector_d rv;
  EXPECT_FLOAT_EQ(0.0,sum(rv));
  rv = row_vector_d(1);
  rv[0] = 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(rv));
  rv = row_vector_d(3);
  rv[0] = 5.0;
  rv[1] = 10.0;
  rv[2] = 100.0;
  EXPECT_FLOAT_EQ(115.0,sum(rv));

  matrix_d m;
  EXPECT_FLOAT_EQ(0.0,sum(m));
  m = matrix_d(1,1);
  m << 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(m));
  m = matrix_d(3,2);
  m << 1, 2, 3, 4, 5, 6;
  EXPECT_FLOAT_EQ(21.0,sum(m));
}

TEST(matrix_test,stdVectorProdInt) {
  using stan::math::prod;
  std::vector<int> v;
  EXPECT_EQ(1,prod(v));
  v.push_back(2);
  EXPECT_EQ(2,prod(v));
  v.push_back(3);
  EXPECT_EQ(6,prod(v));
}
TEST(matrix_test,stdVectorProd) {
  using stan::math::prod;
  std::vector<double> x;
  EXPECT_FLOAT_EQ(1.0,prod(x));
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(2.0,prod(x));
  x.push_back(3);
  EXPECT_FLOAT_EQ(6.0,prod(x));

  vector_d v;
  EXPECT_FLOAT_EQ(1.0,prod(v));
  v = vector_d(1);
  v << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(v));
  v = vector_d(2);
  v << 2.0, 3.0;
  EXPECT_FLOAT_EQ(6.0,prod(v));

  row_vector_d rv;
  EXPECT_FLOAT_EQ(1.0,prod(rv));
  rv = row_vector_d(1);
  rv << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(rv));
  rv = row_vector_d(2);
  rv << 2.0, 3.0;
  EXPECT_FLOAT_EQ(6.0,prod(rv));

  matrix_d m;
  EXPECT_FLOAT_EQ(1.0,prod(m));
  m = matrix_d(1,1);
  m << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(m));
  m = matrix_d(2,3);
  m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  EXPECT_FLOAT_EQ(720.0,prod(m));
}




TEST(matrixTest,eltMultiplyVec) {
  vector_d v1(2);
  vector_d v2(2);
  v1 << 1, 2;
  v2 << 10, 100;
  vector_d v = stan::math::elt_multiply(v1,v2);
  EXPECT_FLOAT_EQ(10.0, v(0));
  EXPECT_FLOAT_EQ(200.0, v(1));
}
TEST(matrixTest,eltMultiplyVecException) {
  vector_d v1(2);
  vector_d v2(3);
  v1 << 1, 2;
  v2 << 10, 100, 1000;
  EXPECT_THROW(stan::math::elt_multiply(v1,v2), std::domain_error);
}
TEST(matrixTest,eltMultiplyRowVec) {
  row_vector_d v1(2);
  row_vector_d v2(2);
  v1 << 1, 2;
  v2 << 10, 100;
  row_vector_d v = stan::math::elt_multiply(v1,v2);
  EXPECT_FLOAT_EQ(10.0, v(0));
  EXPECT_FLOAT_EQ(200.0, v(1));
}
TEST(matrixTest,eltMultiplyRowVecException) {
  row_vector_d v1(2);
  row_vector_d v2(3);
  v1 << 1, 2;
  v2 << 10, 100, 1000;
  EXPECT_THROW(stan::math::elt_multiply(v1,v2), std::domain_error);
}
TEST(matrixTest,eltMultiplyMatrix) {
  matrix_d m1(2,3);
  matrix_d m2(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 10, 100, 1000, 10000, 100000, 1000000;
  matrix_d m = stan::math::elt_multiply(m1,m2);
  
  EXPECT_EQ(2,m.rows());
  EXPECT_EQ(3,m.cols());
  EXPECT_FLOAT_EQ(10.0, m(0,0));
  EXPECT_FLOAT_EQ(200.0, m(0,1));
  EXPECT_FLOAT_EQ(3000.0, m(0,2));
  EXPECT_FLOAT_EQ(40000.0, m(1,0));
  EXPECT_FLOAT_EQ(500000.0, m(1,1));
  EXPECT_FLOAT_EQ(6000000.0, m(1,2));
}
TEST(matrixTest,eltMultiplyMatrixException) {
  matrix_d m1(2,3);
  matrix_d m2(2,4);
  matrix_d m3(4,3);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << -1, -2, -3, -4, -5, -6, -7, -8;
  m3 << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;
  EXPECT_THROW(stan::math::elt_multiply(m1,m2),std::domain_error);
  EXPECT_THROW(stan::math::elt_multiply(m1,m3),std::domain_error);
}

TEST(matrixTest,eltDivideVec) {
  vector_d v1(2);
  vector_d v2(2);
  v1 << 1, 2;
  v2 << 10, 100;
  vector_d v = stan::math::elt_divide(v1,v2);
  EXPECT_FLOAT_EQ(0.1, v(0));
  EXPECT_FLOAT_EQ(0.02, v(1));
}
TEST(matrixTest,eltDivideVecException) {
  vector_d v1(2);
  vector_d v2(3);
  v1 << 1, 2;
  v2 << 10, 100, 1000;
  EXPECT_THROW(stan::math::elt_divide(v1,v2), std::domain_error);
}
TEST(matrixTest,eltDivideRowVec) {
  row_vector_d v1(2);
  row_vector_d v2(2);
  v1 << 1, 2;
  v2 << 10, 100;
  row_vector_d v = stan::math::elt_divide(v1,v2);
  EXPECT_FLOAT_EQ(0.1, v(0));
  EXPECT_FLOAT_EQ(0.02, v(1));
}
TEST(matrixTest,eltDivideRowVecException) {
  row_vector_d v1(2);
  row_vector_d v2(3);
  v1 << 1, 2;
  v2 << 10, 100, 1000;
  EXPECT_THROW(stan::math::elt_divide(v1,v2), std::domain_error);
}
TEST(matrixTest,eltDivideMatrix) {
  matrix_d m1(2,3);
  matrix_d m2(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 10, 100, 1000, 10000, 100000, 1000000;
  matrix_d m = stan::math::elt_divide(m1,m2);
  
  std::cout << m << std::endl;

  EXPECT_EQ(2,m.rows());
  EXPECT_EQ(3,m.cols());
  EXPECT_FLOAT_EQ(0.1, m(0,0));
  EXPECT_FLOAT_EQ(0.02, m(0,1));
  EXPECT_FLOAT_EQ(0.003, m(0,2));
  EXPECT_FLOAT_EQ(0.0004, m(1,0));
  EXPECT_FLOAT_EQ(0.00005, m(1,1));
  EXPECT_FLOAT_EQ(0.000006, m(1,2));
}
TEST(matrixTest,eltDivideMatrixException) {
  matrix_d m1(2,3);
  matrix_d m2(2,4);
  matrix_d m3(4,3);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << -1, -2, -3, -4, -5, -6, -7, -8;
  m3 << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;
  EXPECT_THROW(stan::math::elt_divide(m1,m2),std::domain_error);
  EXPECT_THROW(stan::math::elt_divide(m1,m3),std::domain_error);
}

TEST(matrixTest,mdivide_left_val) {
  using stan::math::mdivide_left;
  matrix_d Ad(2,2);
  matrix_d I;

  Ad << 2.0, 3.0, 
        5.0, 7.0;

  I = mdivide_left(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

TEST(matrixTest,mdivide_right_val) {
  using stan::math::mdivide_right;
  matrix_d Ad(2,2);
  matrix_d I;

  Ad << 2.0, 3.0, 
        5.0, 7.0;

  I = mdivide_right(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

TEST(matrixTest,mdivide_left_tri_val) {
  using stan::math::mdivide_left_tri;
  matrix_d Ad(2,2);
  matrix_d I;

  Ad << 2.0, 0.0, 
        5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);

  Ad << 2.0, 3.0, 
        0.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

TEST(matrixTest,mdivide_right_tri_val) {
  using stan::math::mdivide_right_tri;
  matrix_d Ad(2,2);
  matrix_d I;

  Ad << 2.0, 0.0, 
        5.0, 7.0;

  I = mdivide_right_tri<Eigen::Lower>(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);

  Ad << 2.0, 3.0, 
        0.0, 7.0;

  I = mdivide_right_tri<Eigen::Upper>(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}
TEST(MathMatrix,dot_self) {
  using stan::math::dot_self;

  Eigen::Matrix<double,Eigen::Dynamic,1> v1(1);
  v1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(v1),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,1> v2(2);
  v2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(v2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,1> v3(3);
  v3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(v3),1E-12);

  Eigen::Matrix<double,1,Eigen::Dynamic> rv1(1);
  rv1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(rv1),1E-12);
  Eigen::Matrix<double,1,Eigen::Dynamic> rv2(2);
  rv2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(rv2),1E-12);
  Eigen::Matrix<double,1,Eigen::Dynamic> rv3(3);
  rv3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(rv3),1E-12);

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(m1),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m2(2,1);
  m2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(m2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m3(3,1);
  m3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(m3),1E-12);

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mm2(1,2);
  mm2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(mm2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mm3(1,3);
  mm3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(mm3),1E-12);

}
TEST(MathMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0),1E-12);
  EXPECT_NEAR(9.0,x(1,0),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0),1E-12);
  EXPECT_NEAR(34.0,x(1,0),1E-12);
}
TEST(MathMatrix,softmax) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,1> x(1);
  x << 0.0;
  
  Matrix<double,Dynamic,1> theta = softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0]);

  Matrix<double,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
  Matrix<double,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0]);
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1]);

  Matrix<double,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
  Matrix<double,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0]);
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1]);
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2]);
}
TEST(MathMatrix,dimensionValidation) {
  using stan::math::determinant;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  Matrix<double,Dynamic,Dynamic> x(3,3);
  x << 1, 2, 3, 1, 4, 9, 1, 8, 27;

  ASSERT_FALSE(boost::math::isnan(determinant(x)));

  Matrix<double,Dynamic,Dynamic> xx(3,2);
  xx << 1, 2, 3, 1, 4, 9;
  EXPECT_THROW(stan::math::determinant(xx),std::domain_error);
  
}
TEST(MathMatrix,nonzeroMinMax) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using std::numeric_limits;
  Matrix<double,Dynamic,Dynamic> m;
  Matrix<double,Dynamic,1> v;
  Matrix<double,1,Dynamic> rv;
  EXPECT_EQ(numeric_limits<double>::infinity(),
            stan::math::min(m));
  EXPECT_EQ(numeric_limits<double>::infinity(),
            stan::math::min(v));
  EXPECT_EQ(numeric_limits<double>::infinity(),
            stan::math::min(rv));

  EXPECT_EQ(-numeric_limits<double>::infinity(),
            stan::math::max(m));
  EXPECT_EQ(-numeric_limits<double>::infinity(),
            stan::math::max(v));
  EXPECT_EQ(-numeric_limits<double>::infinity(),
            stan::math::max(rv));

  EXPECT_THROW(stan::math::mean(m), std::domain_error);
  EXPECT_THROW(stan::math::mean(v), std::domain_error);
  EXPECT_THROW(stan::math::mean(rv), std::domain_error);

  Matrix<double,Dynamic,Dynamic> m_nz(2,3);
  Matrix<double,Dynamic,1> v_nz(2);
  Matrix<double,1,Dynamic> rv_nz(3);
  EXPECT_NO_THROW(stan::math::min(m_nz));
  EXPECT_NO_THROW(stan::math::min(v_nz));
  EXPECT_NO_THROW(stan::math::min(rv_nz));

  EXPECT_NO_THROW(stan::math::max(m_nz));
  EXPECT_NO_THROW(stan::math::max(v_nz));
  EXPECT_NO_THROW(stan::math::max(rv_nz));

  EXPECT_NO_THROW(stan::math::mean(m_nz));
  EXPECT_NO_THROW(stan::math::mean(v_nz));
  EXPECT_NO_THROW(stan::math::mean(rv_nz));
}
TEST(MathMatrix,minVectorValues) {
  using stan::math::min;
  std::vector<int> n;
  EXPECT_THROW(min(n),std::domain_error);
  n.push_back(1);
  EXPECT_EQ(1,min(n));
  n.push_back(2);
  EXPECT_EQ(1,min(n));
  n.push_back(0);
  EXPECT_EQ(0,min(n));
  
  std::vector<double> x;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),min(x));
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0,min(x));
  x.push_back(0.0);
  EXPECT_FLOAT_EQ(0.0,min(x));
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(0.0,min(x));
  x.push_back(-10.0);
  EXPECT_FLOAT_EQ(-10.0,min(x));

  vector_d v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),min(v));
  v = vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(1.0,min(v));
  v = vector_d(2);
  v << 1.0, 0.0;
  EXPECT_FLOAT_EQ(0.0,min(v));
  v = vector_d(3);
  v << 1.0, 0.0, 2.0;
  EXPECT_FLOAT_EQ(0.0,min(v));
  v = vector_d(4);
  v << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(-10.0,min(v));

  row_vector_d rv;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),min(rv));
  rv = row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(1.0,min(rv));
  rv = row_vector_d(2);
  rv << 1.0, 0.0;
  EXPECT_FLOAT_EQ(0.0,min(rv));
  rv = row_vector_d(3);
  rv << 1.0, 0.0, 2.0;
  EXPECT_FLOAT_EQ(0.0,min(rv));
  rv = row_vector_d(4);
  rv << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(-10.0,min(rv));

  matrix_d m;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),min(m));
  m = matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(1.0,min(m));
  m = matrix_d(2,2);
  m << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(-10.0,min(m));
}

TEST(MathMatrix,maxVectorValues) {
  using stan::math::max;
  std::vector<int> n;
  EXPECT_THROW(max(n),std::domain_error);
  n.push_back(1);
  EXPECT_EQ(1,max(n));
  n.push_back(2);
  EXPECT_EQ(2,max(n));
  n.push_back(0);
  EXPECT_EQ(2,max(n));
  
  std::vector<double> x;
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(),max(x));
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0,max(x));
  x.push_back(0.0);
  EXPECT_FLOAT_EQ(1.0,max(x));
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(2.0,max(x));
  x.push_back(-10.0);
  EXPECT_FLOAT_EQ(2.0,max(x));

  vector_d v;
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(),max(v));
  v = vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(1.0,max(v));
  v = vector_d(2);
  v << 1.0, 0.0;
  EXPECT_FLOAT_EQ(1.0,max(v));
  v = vector_d(3);
  v << 1.0, 0.0, 2.0;
  EXPECT_FLOAT_EQ(2.0,max(v));
  v = vector_d(4);
  v << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(2.0,max(v));

  row_vector_d rv;
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(),max(rv));
  rv = row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(1.0,max(rv));
  rv = row_vector_d(2);
  rv << 1.0, 0.0;
  EXPECT_FLOAT_EQ(1.0,max(rv));
  rv = row_vector_d(3);
  rv << 1.0, 0.0, 2.0;
  EXPECT_FLOAT_EQ(2.0,max(rv));
  rv = row_vector_d(4);
  rv << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(2.0,max(rv));

  matrix_d m;
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(),max(m));
  m = matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(1.0,max(m));
  m = matrix_d(2,2);
  m << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(2.0,max(m));
}

TEST(MathMatrix,sd) {
  using stan::math::sd;
  std::vector<double> x;
  EXPECT_THROW(sd(x),std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0,sd(x));
  x.push_back(2.0);
  EXPECT_NEAR(0.7071068,sd(x),0.000001);
  x.push_back(3.0);
  EXPECT_FLOAT_EQ(1.0,sd(x));

  vector_d v;
  EXPECT_THROW(sd(v),std::domain_error);
  v = vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(1.0,sd(v));
  v = vector_d(2);
  v << 1.0, 2.0;
  EXPECT_NEAR(0.7071068,sd(v),0.000001);
  v = vector_d(3);
  v << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,sd(v));

  row_vector_d rv;
  EXPECT_THROW(sd(rv),std::domain_error);
  rv = row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(1.0,sd(rv));
  rv = row_vector_d(2);
  rv << 1.0, 2.0;
  EXPECT_NEAR(0.7071068,sd(rv),0.000001);
  rv = row_vector_d(3);
  rv << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,sd(rv));


  matrix_d m;
  EXPECT_THROW(sd(m),std::domain_error);
  m = matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(1.0,sd(m));
  m = matrix_d(2,3);
  m << 1.0, 2.0, 4.0, 9.0, 16.0, 25.0;
  EXPECT_NEAR(9.396808,sd(m),0.000001);
}

TEST(MathMatrix,variance) {
  using stan::math::variance;
  std::vector<double> x;
  EXPECT_THROW(variance(x),std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0,variance(x));
  x.push_back(2.0);
  EXPECT_NEAR(0.5,variance(x),0.000001);
  x.push_back(3.0);
  EXPECT_FLOAT_EQ(1.0,variance(x));

  vector_d v;
  EXPECT_THROW(variance(v),std::domain_error);
  v = vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(1.0,variance(v));
  v = vector_d(2);
  v << 1.0, 2.0;
  EXPECT_NEAR(0.5,variance(v),0.000001);
  v = vector_d(3);
  v << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,variance(v));

  row_vector_d rv;
  EXPECT_THROW(variance(rv),std::domain_error);
  rv = row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(1.0,variance(rv));
  rv = row_vector_d(2);
  rv << 1.0, 2.0;
  EXPECT_NEAR(0.5,variance(rv),0.000001);
  rv = row_vector_d(3);
  rv << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,variance(rv));


  matrix_d m;
  EXPECT_THROW(variance(m),std::domain_error);
  m = matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(1.0,variance(m));
  m = matrix_d(2,3);
  m << 1.0, 2.0, 4.0, 9.0, 16.0, 25.0;
  EXPECT_NEAR(88.3,variance(m),0.000001);
}

TEST(MathMatrix,mean) {
  using stan::math::mean;
  std::vector<double> x;
  EXPECT_THROW(mean(x),std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0,mean(x));
  x.push_back(2.0);
  EXPECT_NEAR(1.5,mean(x),0.000001);
  x.push_back(3.0);
  EXPECT_FLOAT_EQ(2.0,mean(x));

  vector_d v;
  EXPECT_THROW(mean(v),std::domain_error);
  v = vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(1.0,mean(v));
  v = vector_d(2);
  v << 1.0, 2.0;
  EXPECT_NEAR(1.5,mean(v),0.000001);
  v = vector_d(3);
  v << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(2.0,mean(v));

  row_vector_d rv;
  EXPECT_THROW(mean(rv),std::domain_error);
  rv = row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(1.0,mean(rv));
  rv = row_vector_d(2);
  rv << 1.0, 2.0;
  EXPECT_NEAR(1.5,mean(rv),0.000001);
  rv = row_vector_d(3);
  rv << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(2.0,mean(rv));

  matrix_d m;
  EXPECT_THROW(mean(m),std::domain_error);
  m = matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(1.0,mean(m));
  m = matrix_d(2,3);
  m << 1.0, 2.0, 4.0, 9.0, 16.0, 25.0;
  EXPECT_FLOAT_EQ(9.5,mean(m));
}
