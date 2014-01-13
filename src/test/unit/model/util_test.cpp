#include <stan/model/util.hpp>
#include <gtest/gtest.h>

#include <vector>
#include <stan/io/reader.hpp>
#include <stan/math/matrix/accumulator.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>
#include <stan/prob/transform.hpp>

class TestModel_uniform_01 {
public:
  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(std::vector<T__>& params_r__,
               std::vector<int>& params_i__,
               std::ostream* pstream__ = 0) const {
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    
    // model parameters
    stan::io::reader<T__> in__(params_r__,params_i__);
    
    T__ y;
    if (jacobian__)
      y = in__.scalar_lub_constrain(0,1,lp__);
    else
      y = in__.scalar_lub_constrain(0,1);
    
    lp_accum__.add(stan::prob::uniform_log<propto__>(y, 0, 1));
    lp_accum__.add(lp__);

    return lp_accum__.sum();
  }
};

TEST(ModelUtil, finite_diff_grad__false_false) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  
  for (int i = 0; i < 10; i++) {
    params_r[0] = (i-5.0) * 10;
    
    stan::model::finite_diff_grad<false,false,TestModel_uniform_01>
      (model, params_r, params_i, gradient);
    
    ASSERT_EQ(1U, gradient.size());
    EXPECT_FLOAT_EQ(0.0, gradient[0]);
  }
}
TEST(ModelUtil, finite_diff_grad__false_true) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  
  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<false,true,TestModel_uniform_01>
      (model, params_r, params_i, gradient);
    
    ASSERT_EQ(1U, gradient.size());
    
    // derivative of the transform
    double expected_gradient = -std::tanh(0.5 * x);    
    EXPECT_FLOAT_EQ(expected_gradient, gradient[0]);
  }
}

TEST(ModelUtil, finite_diff_grad__true_false) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  
  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<true,false,TestModel_uniform_01>
      (model, params_r, params_i, gradient);
    
    ASSERT_EQ(1U, gradient.size());
    
    EXPECT_FLOAT_EQ(0.0, gradient[0]);
  }
}

TEST(ModelUtil, finite_diff_grad__true_true) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  
  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<true,true,TestModel_uniform_01>
      (model, params_r, params_i, gradient);
    
    ASSERT_EQ(1U, gradient.size());
    
    double expected_gradient = -std::tanh(0.5 * x);    
    EXPECT_FLOAT_EQ(expected_gradient, gradient[0]);
  }
}
