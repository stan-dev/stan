#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(AgradMixMatrix,value_of_rec) {
  using stan::math::var;
  using stan::math::fvar;
  using stan::math::value_of_rec;
  using std::vector;

  vector<double> a_vals;

  for (size_t i = 0; i < 10; ++i)
    a_vals.push_back(i + 1);

  vector<double> b_vals;

  for (size_t i = 10; i < 15; ++i)
    b_vals.push_back(i + 1);
  
  Eigen::Matrix<fvar<var>,2,5> fv_a;
  fill(a_vals, fv_a);
  Eigen::Matrix<fvar<var>,5,1> fv_b;
  fill(b_vals, fv_b);

  Eigen::Matrix<fvar<fvar<var> >,2,5> ffv_a;
  fill(a_vals, ffv_a);
  Eigen::Matrix<fvar<fvar<var> >,5,1> ffv_b;
  fill(b_vals, ffv_b);

  Eigen::MatrixXd d_fv_a = value_of_rec(fv_a);
  Eigen::MatrixXd d_fv_b = value_of_rec(fv_b);
  Eigen::MatrixXd d_ffv_a = value_of_rec(ffv_a);
  Eigen::MatrixXd d_ffv_b = value_of_rec(ffv_b);

  for (size_type i = 0; i < 5; ++i){
    EXPECT_FLOAT_EQ(b_vals[i], d_fv_b(i));
    EXPECT_FLOAT_EQ(b_vals[i], d_ffv_b(i));
  }

  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 5; ++j){
      EXPECT_FLOAT_EQ(a_vals[j * 2 + i], d_fv_a(i,j));
      EXPECT_FLOAT_EQ(a_vals[j * 2 + i], d_ffv_a(i,j));   
    }
}
