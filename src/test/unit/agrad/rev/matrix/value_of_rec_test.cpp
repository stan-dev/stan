#include <stan/agrad/rev/matrix/value_of_rec.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradMatrix,value_of_rec) {
  using stan::agrad::var;
  using stan::agrad::value_of_rec;
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

  Eigen::Matrix<var,2,5> v_a;
  fill(a_vals, v_a);
  Eigen::Matrix<var,5,1> v_b;
  fill(b_vals, v_b);

  Eigen::MatrixXd d_v_a = value_of_rec(v_a);
  Eigen::MatrixXd d_v_b = value_of_rec(v_b);

  for (size_type i = 0; i < 5; ++i){
    EXPECT_FLOAT_EQ(b(i), d_v_b(i));
  }

  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 5; ++j){
      EXPECT_FLOAT_EQ(a(i,j), d_v_a(i,j));
    }
}
