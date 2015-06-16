#include <stan/math/prim/mat/fun/num_elements.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, num_elements) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using std::vector;
  using stan::math::num_elements;

  Matrix<double, Dynamic, Dynamic> a1(3,2);
  EXPECT_EQ(num_elements(a1), 6);

  Matrix<double, Dynamic, Dynamic> a2;
  EXPECT_EQ(num_elements(a2), 0);
  
  Matrix<double, 1, Dynamic> a3;
  EXPECT_EQ(num_elements(a3), 0);
  
  Matrix<double, Dynamic, 1> a4;
  EXPECT_EQ(num_elements(a4), 0);
  
  Matrix<double, 1, Dynamic> a5(7);
  EXPECT_EQ(num_elements(a5), 7);
  
  Matrix<double, Dynamic, 1> a6(9);
  EXPECT_EQ(num_elements(a6), 9);
  
  Matrix<double, Dynamic, Dynamic> a7(3,0);
  EXPECT_EQ(num_elements(a7), 0);
  
  Matrix<double, Dynamic, Dynamic> a8(0,3);
  EXPECT_EQ(num_elements(a8), 0);
  
  vector<double> vec0;
  EXPECT_EQ(num_elements(vec0), 0);
  
  vec0.push_back(3.0);
  EXPECT_EQ(num_elements(vec0), 1);  
  
  vector<vector <double> > vec1(3, vector<double>(2));
  EXPECT_EQ(num_elements(vec1), 6);
  
  vector<vector <double> > vec2(3, vector<double>(0));
  EXPECT_EQ(num_elements(vec2), 0);
  
  vector<vector <double> > vec3(0, vector<double>(3));
  EXPECT_EQ(num_elements(vec3), 0);
  
  vector<vector <double> > vec4;
  EXPECT_EQ(num_elements(vec4), 0);
  
  vector<vector<vector<double> > > vec5(3, vector<vector<double> >(2, vector <double>(4)));
  EXPECT_EQ(num_elements(vec5), 24);
  
  vector<vector<vector<double> > > vec6(3, vector<vector<double> >(0, vector <double>(4)));
  EXPECT_EQ(num_elements(vec6), 0);
  
  vector<vector<vector<double> > > vec7(0, vector<vector<double> >(2, vector <double>(4)));
  EXPECT_EQ(num_elements(vec7), 0);
  
  vector<vector<vector<double> > > vec8(3, vector<vector<double> >(2, vector <double>(0)));
  EXPECT_EQ(num_elements(vec8), 0);
  
  vector<vector<vector<double> > > vec9;
  EXPECT_EQ(num_elements(vec9), 0);
  
  vector<vector<vector<double> > > vec10(10);
  EXPECT_EQ(num_elements(vec10), 0);
  
  vector<vector<vector<double> > > vec11(10, vector<vector<double> >(15));
  EXPECT_EQ(num_elements(vec11), 0);
  
  vector<vector<vector<vector<double> > > > vec12(3, vector<vector<vector <double> > >(2, vector<vector <double> >(4, vector<double>(5))));
  EXPECT_EQ(num_elements(vec12), 24*5);
  
  vector<Matrix<double, Dynamic, Dynamic> > mixed1(5, Matrix<double, Dynamic, Dynamic>(7, 2));
  EXPECT_EQ(num_elements(mixed1), 5*7*2);
  
  vector<Matrix<double, Dynamic, Dynamic> > mixed2(0, Matrix<double, Dynamic, Dynamic>(7, 2));
  EXPECT_EQ(num_elements(mixed2), 0);
  
  vector<Matrix<double, Dynamic, Dynamic> > mixed3(5, Matrix<double, Dynamic, Dynamic>(0, 2));
  EXPECT_EQ(num_elements(mixed3), 0);
  
  vector<Matrix<double, Dynamic, Dynamic> > mixed4(5, Matrix<double, Dynamic, Dynamic>(7, 0));
  EXPECT_EQ(num_elements(mixed4), 0);
  
  vector<Matrix<double, Dynamic, Dynamic> > mixed5(0, Matrix<double, Dynamic, Dynamic>(0, 2));
  EXPECT_EQ(num_elements(mixed5), 0);
  
  vector<Matrix<double, Dynamic, Dynamic> > mixed6(0, Matrix<double, Dynamic, Dynamic>(7, 0));
  EXPECT_EQ(num_elements(mixed6), 0);
  
  vector<Matrix<double, Dynamic, Dynamic> > mixed7(5, Matrix<double, Dynamic, Dynamic>(0, 0));
  EXPECT_EQ(num_elements(mixed7), 0);
  
  vector<Matrix<double, Dynamic, Dynamic> > mixed8;
  EXPECT_EQ(num_elements(mixed8), 0);
  
  vector<vector<Matrix<double, Dynamic, Dynamic> > > mixed9(5, vector<Matrix<double, Dynamic, Dynamic> >(4, Matrix<double, Dynamic, Dynamic>(7, 2)));
  EXPECT_EQ(num_elements(mixed9), 5*4*7*2);
  
  vector<vector<Matrix<double, Dynamic, Dynamic> > > mixed10(0, vector<Matrix<double, Dynamic, Dynamic> >(4, Matrix<double, Dynamic, Dynamic>(7, 2)));
  EXPECT_EQ(num_elements(mixed10), 0);
  
  vector<vector<Matrix<double, Dynamic, Dynamic> > > mixed11(5, vector<Matrix<double, Dynamic, Dynamic> >(0, Matrix<double, Dynamic, Dynamic>(7, 2)));
  EXPECT_EQ(num_elements(mixed11), 0);
  
  vector<vector<Matrix<double, Dynamic, Dynamic> > > mixed12(5, vector<Matrix<double, Dynamic, Dynamic> >(4, Matrix<double, Dynamic, Dynamic>(0, 2)));
  EXPECT_EQ(num_elements(mixed12), 0);
  
  vector<vector<Matrix<double, Dynamic, Dynamic> > > mixed13(5, vector<Matrix<double, Dynamic, Dynamic> >(4, Matrix<double, Dynamic, Dynamic>(7, 0)));
  EXPECT_EQ(num_elements(mixed13), 0);
  
  vector<vector<Matrix<double, Dynamic, Dynamic> > > mixed14;
  EXPECT_EQ(num_elements(mixed14), 0);
}
