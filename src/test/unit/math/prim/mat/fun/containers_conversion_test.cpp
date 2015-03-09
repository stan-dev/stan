#include <stan/math/prim/mat/fun/to_matrix.hpp>
#include <stan/math/prim/mat/fun/to_vector.hpp>
#include <stan/math/prim/mat/fun/to_row_vector.hpp>
#include <stan/math/prim/mat/fun/to_array_2d.hpp>
#include <stan/math/prim/mat/fun/to_array_1d.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using stan::math::to_matrix;
using stan::math::to_vector;
using stan::math::to_row_vector;
using stan::math::to_array_2d;
using stan::math::to_array_1d;  
using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;

TEST(MathMatrix, conversions_1) {

  Matrix<double, Dynamic, Dynamic> a1(3,2);
  a1 << 1.1, 2.53,
        3.98, 4.1,
        5.1, 6.87;

  Matrix<double, Dynamic, Dynamic> a2;
  
  vector< vector <double> > b1(3, vector <double>(2));
  b1[0][0] = 11.1;
  b1[0][1] = 12.7;
  b1[1][0] = 13.53;
  b1[1][1] = 14.1;
  b1[2][0] = 15;
  b1[2][1] = 16.5;
  
  vector< vector <double> > b2;

  Matrix<double, Dynamic, 1> c1(3);
  c1 << 21.0, 22.1, 23.53;

  Matrix<double, Dynamic, 1> c2;


  Matrix<double, 1, Dynamic> d1(3);
  d1 << 31.3, 32.53, 33;

  Matrix<double, 1, Dynamic> d2;

  vector<double> e1(3);
  e1[0] = 41.1;
  e1[1] = 42.45;
  e1[2] = 43.53;

  vector<double> e2;
  
  vector< vector <int> > f1(3, vector <int>(2));
  f1[0][0] = 53;
  f1[0][1] = 54;
  f1[1][0] = 55;
  f1[1][1] = 56;
  f1[2][0] = 57;
  f1[2][1] = 58;

  vector< vector <int> > f2;

  vector<int> g1(3);
  g1[0] = 61;
  g1[1] = 62;
  g1[2] = 63;
  
  vector<int> g2;
  
  
  //Tests for empty containers
  //matrix to_matrix(vector)
  EXPECT_NO_THROW(to_matrix(a2));
  
  //matrix to_matrix(vector)
  EXPECT_NO_THROW(to_matrix(c2));
  
  //matrix to_matrix(row_vector)
  EXPECT_NO_THROW(to_matrix(d2));
  
  //matrix to_matrix(real[,])
  EXPECT_NO_THROW(to_matrix(b2));
  
  //matrix to_matrix(int[,])
  EXPECT_NO_THROW(to_matrix(f2));
  
  //vector to_vector(matrix)
  EXPECT_NO_THROW(to_vector(a2));
  
  //vector to_vector(row_vector)
  EXPECT_NO_THROW(to_vector(d2));
  
  //vector to_vector(vector)
  EXPECT_NO_THROW(to_vector(c2));
  
  //vector to_vector(real[])
  EXPECT_NO_THROW(to_vector(e2));
  
  //vector to_vector(int[])
  EXPECT_NO_THROW(to_vector(g2));
  
  //row_vector to_row_vector(matrix)
  EXPECT_NO_THROW(to_row_vector(a2));
  
  //row_vector to_row_vector(vector)
  EXPECT_NO_THROW(to_row_vector(c2));
  
  //row_vector to_row_vector(row_vector)
  EXPECT_NO_THROW(to_row_vector(d2));
  
  //row_vector to_row_vector(real[])
  EXPECT_NO_THROW(to_row_vector(e2));
  
  //row_vector to_row_vector(int[])
  EXPECT_NO_THROW(to_row_vector(g2));
  
  //real[,] to_array_2d(matrix)
  EXPECT_NO_THROW(to_array_2d(a2));
  
  //real[] to_array_1d(matrix)
  EXPECT_NO_THROW(to_array_1d(a2));
  
  //real[] to_array_1d(row_vector)
  EXPECT_NO_THROW(to_array_1d(d2));
  
  //real[] to_array_1d(vector)
  EXPECT_NO_THROW(to_array_1d(c2));
  

  //matrix to_matrix(vector)
  a2 = to_matrix(a1);
  expect_matrix_eq(a1, a2);
  
  //matrix to_matrix(vector)
  a2 = to_matrix(c1);
  EXPECT_EQ(a2(0, 0), c1(0));
  EXPECT_EQ(a2(1, 0), c1(1));
  EXPECT_EQ(a2(2, 0), c1(2));
  
  //matrix to_matrix(row_vector)
  a2 = to_matrix(d1);
  EXPECT_EQ(a2(0, 0), d1(0));
  EXPECT_EQ(a2(0, 1), d1(1));
  EXPECT_EQ(a2(0, 2), d1(2));

  //matrix to_matrix(real[,])
  a2 = to_matrix(b1);

  EXPECT_EQ(a2(0, 0), b1[0][0]);
  EXPECT_EQ(a2(0, 1), b1[0][1]);
  EXPECT_EQ(a2(1, 0), b1[1][0]);
  EXPECT_EQ(a2(1, 1), b1[1][1]);
  EXPECT_EQ(a2(2, 0), b1[2][0]);
  EXPECT_EQ(a2(2, 1), b1[2][1]);
  
  //matrix to_matrix(int[,])
  a2 = to_matrix(f1);

  EXPECT_EQ(a2(0, 0), f1[0][0]);
  EXPECT_EQ(a2(0, 1), f1[0][1]);
  EXPECT_EQ(a2(1, 0), f1[1][0]);
  EXPECT_EQ(a2(1, 1), f1[1][1]);
  EXPECT_EQ(a2(2, 0), f1[2][0]);
  EXPECT_EQ(a2(2, 1), f1[2][1]);
  
  //vector to_vector(matrix)
  c2 = to_vector(a1);
  EXPECT_EQ(c2(0), a1(0, 0));
  EXPECT_EQ(c2(1), a1(1, 0));
  EXPECT_EQ(c2(2), a1(2, 0));
  EXPECT_EQ(c2(3), a1(0, 1));
  EXPECT_EQ(c2(4), a1(1, 1));
  EXPECT_EQ(c2(5), a1(2, 1));
  
  //vector to_vector(row_vector)
  c2 = to_vector(d1);
  for(int i=0; i<3; i++)
    EXPECT_EQ(c2(i), d1(i));

  //vector to_vector(vector)
  c2 = to_vector(c1);
  expect_matrix_eq(c1, c2);
    
  //vector to_vector(real[])
  c2 = to_vector(e1);
  
  EXPECT_EQ(c2(0), e1[0]);
  EXPECT_EQ(c2(1), e1[1]);
  EXPECT_EQ(c2(2), e1[2]);
  
  //vector to_vector(int[])
  c2 = to_vector(g1);
  
  EXPECT_EQ(c2(0), g1[0]);
  EXPECT_EQ(c2(1), g1[1]);
  EXPECT_EQ(c2(2), g1[2]);
  
  //row_vector to_row_vector(matrix)
  d2 = to_row_vector(a1);
  EXPECT_EQ(d2(0), a1(0, 0));
  EXPECT_EQ(d2(1), a1(1, 0));
  EXPECT_EQ(d2(2), a1(2, 0));
  EXPECT_EQ(d2(3), a1(0, 1));
  EXPECT_EQ(d2(4), a1(1, 1));
  EXPECT_EQ(d2(5), a1(2, 1));
  
  //row_vector to_row_vector(vector)
  d2 = to_row_vector(c1);
  for(int i=0; i<3; i++)
    EXPECT_EQ(d2(i), c1(i));

  //row_vector to_row_vector(row_vector)
  d2 = to_row_vector(d1);
  expect_matrix_eq(d1, d2);

  //row_vector to_row_vector(real[])
  d2 = to_row_vector(e1);
  
  EXPECT_EQ(d2(0), e1[0]);
  EXPECT_EQ(d2(1), e1[1]);
  EXPECT_EQ(d2(2), e1[2]);
  
  //row_vector to_row_vector(int[])
  d2 = to_row_vector(g1);

  EXPECT_EQ(d2(0), g1[0]);
  EXPECT_EQ(d2(1), g1[1]);
  EXPECT_EQ(d2(2), g1[2]);

  //real[,] to_array_2d(matrix)
  b2 = to_array_2d(a1);

  EXPECT_EQ(a1(0, 0), b2[0][0]);
  EXPECT_EQ(a1(0, 1), b2[0][1]);
  EXPECT_EQ(a1(1, 0), b2[1][0]);
  EXPECT_EQ(a1(1, 1), b2[1][1]);
  EXPECT_EQ(a1(2, 0), b2[2][0]);
  EXPECT_EQ(a1(2, 1), b2[2][1]);

  //real[] to_array_1d(matrix)
  e2 = to_array_1d(a1);
  
  EXPECT_EQ(a1(0, 0), e2[0]);
  EXPECT_EQ(a1(1, 0), e2[1]);
  EXPECT_EQ(a1(2, 0), e2[2]);
  EXPECT_EQ(a1(0, 1), e2[3]);
  EXPECT_EQ(a1(1, 1), e2[4]);
  EXPECT_EQ(a1(2, 1), e2[5]);

  //real[] to_array_1d(row_vector)
  e2 = to_array_1d(d1);
  EXPECT_EQ(d1(0), e2[0]);
  EXPECT_EQ(d1(1), e2[1]);
  EXPECT_EQ(d1(2), e2[2]);

  //real[] to_array_1d(vector)
  e2 = to_array_1d(c1);
  EXPECT_EQ(c1(0), e2[0]);
  EXPECT_EQ(c1(1), e2[1]);
  EXPECT_EQ(c1(2), e2[2]);
  
  //Now we play with some lossless operations
  expect_matrix_eq(a1, to_matrix(to_array_2d(a1)));
  expect_matrix_eq(c1, to_vector(to_array_1d(c1)));
  expect_matrix_eq(c1, to_vector(to_matrix(c1)));
  expect_matrix_eq(c1, to_vector(to_row_vector(c1)));
  expect_matrix_eq(d1, to_row_vector(to_array_1d(d1)));
  expect_matrix_eq(d1, to_row_vector(to_matrix(d1)));
  expect_matrix_eq(d1, to_row_vector(to_vector(d1)));
}
TEST(MathMatrix, conversions_2) {

  vector< vector < vector <double> > > a1(3, vector < vector<double> >(2, vector <double>(4)));
  a1[0][0][0] = 11.341;
  a1[0][1][0] = 12.734;
  a1[1][0][0] = 13.5433;
  a1[1][1][0] = 14.1124;
  a1[2][0][0] = 15;
  a1[2][1][0] = 16.456;
  a1[0][0][1] = 11.5545;
  a1[0][1][1] = 12.45437;
  a1[1][0][1] = 13.3453;
  a1[1][1][1] = 14.134;
  a1[2][0][1] = 15.86;
  a1[2][1][1] = 16.455;
  a1[0][0][2] = 11.451;
  a1[0][1][2] = 12.4537;
  a1[1][0][2] = 13.53453;
  a1[1][1][2] = 14.45431;
  a1[2][0][2] = 15.8556;
  a1[2][1][2] = 16.56545;
  a1[0][0][3] = 11.6541;
  a1[0][1][3] = 12.2237;
  a1[1][0][3] = 13.5453;
  a1[1][1][3] = 14.3451;
  a1[2][0][3] = 15.7867;
  a1[2][1][3] = 16.445;
    
  vector< vector < vector <double> > > a2;
  
  vector< vector <double> > b1(3, vector <double>(2));
  b1[0][0] = 21.1;
  b1[0][1] = 22.7;
  b1[1][0] = 23.53;
  b1[1][1] = 24.1;
  b1[2][0] = 25;
  b1[2][1] = 26.5;
  
  vector< vector <double> > b2;

  vector<double> c1(3);
  c1[0] = 31.1;
  c1[1] = 32.45;
  c1[2] = 33.53;

  vector<double> c2;
  
  vector< vector < vector <int> > > d2;

  vector< vector <int> > e1(3, vector <int>(2));
  e1[0][0] = 53;
  e1[0][1] = 54;
  e1[1][0] = 55;
  e1[1][1] = 56;
  e1[2][0] = 57;
  e1[2][1] = 58;

  vector< vector <int> > e2;

  vector<int> f1(3);
  f1[0] = 61;
  f1[1] = 62;
  f1[2] = 63;
  
  vector<int> f2;
  
  EXPECT_NO_THROW(to_array_1d(a2));
  EXPECT_NO_THROW(to_array_1d(b2));
  EXPECT_NO_THROW(to_array_1d(c2));
  EXPECT_NO_THROW(to_array_1d(d2));
  EXPECT_NO_THROW(to_array_1d(e2));
  EXPECT_NO_THROW(to_array_1d(f2));
  
  c2 = to_array_1d(a1);
  for (size_t i = 0, ijk = 0; i < 3; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++, ijk++)
        EXPECT_EQ(a1[i][j][k], c2[ijk]);
  
  c2 = to_array_1d(b1);
  EXPECT_EQ(b1[0][0], c2[0]);
  EXPECT_EQ(b1[0][1], c2[1]);
  EXPECT_EQ(b1[1][0], c2[2]);
  EXPECT_EQ(b1[1][1], c2[3]);
  EXPECT_EQ(b1[2][0], c2[4]);
  EXPECT_EQ(b1[2][1], c2[5]);

  c2 = to_array_1d(c1);
  EXPECT_EQ(c1[0], c2[0]);
  EXPECT_EQ(c1[1], c2[1]);
  EXPECT_EQ(c1[2], c2[2]);
}
