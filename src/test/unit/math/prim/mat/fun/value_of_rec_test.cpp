#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>

template<typename T, int R, int C>
void fill(const std::vector<double>& contents,
          Eigen::Matrix<T,R,C>& M){
  size_t ij = 0;
  for (int j = 0; j < C; ++j)
    for (int i = 0; i < R; ++i)
      M(i,j) = T(contents[ij++]);
}

TEST(MathMatrix,value_of_rec) {
  using stan::math::value_of_rec;
  using std::vector;

  vector<double> a_vals;

  for (size_t i = 0; i < 10; ++i)
    a_vals.push_back(i + 1);

  vector<double> b_vals;

  for (size_t i = 10; i < 15; ++i)
    b_vals.push_back(i + 1);
  
  Eigen::Matrix<double,2,5> a; 
  fill(a_vals, a);
  Eigen::Matrix<double,5,1> b;
  fill(b_vals, b);

  Eigen::MatrixXd d_a = value_of_rec(a);
  Eigen::VectorXd d_b = value_of_rec(b);

  for (int i = 0; i < 5; ++i)
    EXPECT_FLOAT_EQ(b(i), d_b(i));

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 5; ++j)
      EXPECT_FLOAT_EQ(a(i,j), d_a(i,j));
}
