#include <stan/agrad/rev/matrix/crossprod.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>

void test_crossprod(const stan::agrad::matrix_v& L) {
  using stan::agrad::matrix_v;
  using stan::agrad::crossprod;
  matrix_v LLT_eigen = L.transpose() * L;
  matrix_v LLT_stan = crossprod(L);
  EXPECT_EQ(L.rows(),LLT_stan.rows());
  EXPECT_EQ(L.cols(),LLT_stan.cols());
  for (int m = 0; m < L.rows(); ++m)
    for (int n = 0; n < L.cols(); ++n)
      EXPECT_FLOAT_EQ(LLT_eigen(m,n).val(), LLT_stan(m,n).val());
}

TEST(AgradRevMatrix, crossprod) {
  using stan::agrad::matrix_v;

  matrix_v L(3,3);
  L << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  test_crossprod(L);
  //  test_tcrossprod_grad(L, L.rows(), L.cols());

  matrix_v I(2,2);
  I << 3, 0,
    4, -3;
  test_crossprod(I);
  //  test_tcrossprod_grad(I, I.rows(), I.cols());

  matrix_v J(1,1);
  J << 3.0;
  test_crossprod(J);
  //  test_tcrossprod_grad(J, J.rows(), J.cols());

  matrix_v K(0,0);
  test_crossprod(K);
  //  test_tcrossprod_grad(K, K.rows(), K.cols());

}
