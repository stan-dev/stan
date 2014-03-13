#include <stdexcept>
#include <stan/math/matrix/flatten.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, flatten) {
  
  using stan::math::flatten;  
  using std::vector;

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
  
  EXPECT_NO_THROW(flatten(a2));
  EXPECT_NO_THROW(flatten(b2));
  EXPECT_NO_THROW(flatten(c2));
  EXPECT_NO_THROW(flatten(d2));
  EXPECT_NO_THROW(flatten(e2));
  EXPECT_NO_THROW(flatten(f2));
  
  c2 = flatten(a1);
  for (size_t i = 0, ijk; i < 3; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++, ijk++)
        EXPECT_EQ(a1[i][j][k], c2[ijk]);
  
  c2 = flatten(b1);
  EXPECT_EQ(b1[0][0], c2[0]);
  EXPECT_EQ(b1[0][1], c2[1]);
  EXPECT_EQ(b1[1][0], c2[2]);
  EXPECT_EQ(b1[1][1], c2[3]);
  EXPECT_EQ(b1[2][0], c2[4]);
  EXPECT_EQ(b1[2][1], c2[5]);

  c2 = flatten(c1);
  EXPECT_EQ(c1[0], c2[0]);
  EXPECT_EQ(c1[1], c2[1]);
  EXPECT_EQ(c1[2], c2[2]);
}
