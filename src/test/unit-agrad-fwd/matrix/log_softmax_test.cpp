#include <stdexcept>
#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/fwd/matrix/log_softmax.hpp>
#include <stan/agrad/fwd/matrix/softmax.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

void test_log_softmax(Eigen::Matrix<stan::agrad::fvar<double>,Eigen::Dynamic,1>& v) {
  using stan::math::softmax;
  using stan::math::log_softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;

  for (int i = 0; i < v.size(); ++i) {

    for (int k = 0; k < v.size(); ++k)
      v(k).d_ = i == k ? 2.0 : 0;

    Matrix<fvar<double>,Dynamic,1> softmax_v_expected = softmax(v);
    Matrix<fvar<double>,Dynamic,1> log_softmax_v_expected(v.size());
    for (int k = 0; k < v.size(); ++k)
      log_softmax_v_expected(k) = log(softmax_v_expected(k));
    
    Matrix<fvar<double>,Dynamic,1> log_softmax_v = log_softmax(v);

    EXPECT_EQ(log_softmax_v_expected.size(), log_softmax_v.size());

    for (int k = 0; k < v.size(); ++k)
      EXPECT_FLOAT_EQ(log_softmax_v_expected(k).val_,
                      log_softmax_v(k).val_);

    for (int k = 0; k < v.size(); ++k)
      EXPECT_FLOAT_EQ(log_softmax_v_expected(k).d_,
                      log_softmax_v(k).d_);
  }
}

TEST(AgradFwdMatrix,logSoftmax) {
  using stan::math::log_softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;

  EXPECT_THROW(log_softmax(vector_fv()),std::domain_error);
  
  Matrix<fvar<double>,Dynamic,1> x(1);
  x << 0.0;
  test_log_softmax(x);

  Matrix<fvar<double>,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
  test_log_softmax(x2);

  Matrix<fvar<double>,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
  test_log_softmax(x3);
}
