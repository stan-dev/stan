#include <stan/math/prim/mat/fun/value_of.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(AgradMixMatrix,value_of) {
  using stan::math::var;
  using stan::math::fvar;
  using stan::math::value_of;
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

  Eigen::Matrix<fvar<var>,2,5> fv_a;
  fill(a_vals, fv_a);
  Eigen::Matrix<fvar<var>,5,1> fv_b;
  fill(b_vals, fv_b);

  Eigen::Matrix<fvar<fvar<var> >,2,5> ffv_a;
  fill(a_vals, ffv_a);
  Eigen::Matrix<fvar<fvar<var> >,5,1> ffv_b;
  fill(b_vals, ffv_b);

  Eigen::MatrixXd d_a = value_of(a);
  Eigen::VectorXd d_b = value_of(b);
  Eigen::MatrixXd d_v_a = value_of(v_a);
  Eigen::MatrixXd d_v_b = value_of(v_b);
  Eigen::Matrix<var,-1,-1> d_fv_a = value_of(fv_a);
  Eigen::Matrix<var,-1,-1> d_fv_b = value_of(fv_b);
  Eigen::Matrix<fvar<var>,-1,-1> d_ffv_a = value_of(ffv_a);
  Eigen::Matrix<fvar<var>,-1,-1> d_ffv_b = value_of(ffv_b);

  for (size_type i = 0; i < 5; ++i){
    EXPECT_FLOAT_EQ(b(i), d_b(i));
    EXPECT_FLOAT_EQ(b(i), d_v_b(i));
    EXPECT_FLOAT_EQ(b(i), d_fv_b(i).val());
    EXPECT_FLOAT_EQ(b(i), d_ffv_b(i).val_.val());
  }

  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 5; ++j){
      EXPECT_FLOAT_EQ(a(i,j), d_a(i,j));
      EXPECT_FLOAT_EQ(a(i,j), d_v_a(i,j));
      EXPECT_FLOAT_EQ(a(i,j), d_fv_a(i,j).val());
      EXPECT_FLOAT_EQ(a(i,j), d_ffv_a(i,j).val_.val());
    }
}
