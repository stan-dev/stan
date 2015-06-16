#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/tcrossprod.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/rev/mat/fun/jacobian.hpp>
#include <stan/math/rev/core.hpp>

void test_tcrossprod(const stan::math::matrix_v& L) {
  using stan::math::matrix_v;
  using stan::math::tcrossprod;
  matrix_v LLT_eigen = L * L.transpose();
  matrix_v LLT_stan = tcrossprod(L);
  EXPECT_EQ(LLT_eigen.rows(),LLT_stan.rows());
  EXPECT_EQ(LLT_eigen.rows(),LLT_stan.cols());
  for (int m = 0; m < LLT_eigen.rows(); ++m)
    for (int n = 0; n < LLT_eigen.cols(); ++n)
      EXPECT_FLOAT_EQ(LLT_eigen(m,n).val(), LLT_stan(m,n).val());
}
TEST(AgradRevMatrix, tcrossprod) {
  using stan::math::matrix_v;

  matrix_v L(3,3);
  L << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  test_tcrossprod(L);

  matrix_v I(2,2);
  I << 3, 0,
    4, -3;
  test_tcrossprod(I);

  matrix_v J(1,1);
  J << 3.0;
  test_tcrossprod(J);

  matrix_v K(0,0);
  test_tcrossprod(K);

  matrix_v M(3,3);
  M << 1, 2, 3,
    1, 4, 9,
    1, 8, 27;
  test_tcrossprod(M);

  matrix_v N(1,3);
  N << 1, 2, 3;
  test_tcrossprod(N);

  matrix_v P(2,3);
  P << 1, 2, 3,
    -1, 4, -9;
  test_tcrossprod(P);

  matrix_v Q(3,2);
  Q << 1, 2, 3,
    -1, 4, -9;
  test_tcrossprod(Q);
}
TEST(AgradRevMatrix, tcrossprodGrad1) {
  using stan::math::tcrossprod;
  using stan::math::matrix_v;

  matrix_v L(1,1);
  L << 3.0;
  AVEC x(1);
  x[0] = L(0,0);

  matrix_v LLt = tcrossprod(L);
  AVEC y(1);
  y[0] = LLt(0,0);

  EXPECT_FLOAT_EQ(9.0, LLt(0,0).val());

  std::vector<VEC > J;
  stan::math::jacobian(y,x,J);

  EXPECT_FLOAT_EQ(6.0, J[0][0]);
}

TEST(AgradRevMatrix, tcrossprodGrad2) {
  using stan::math::tcrossprod;
  using stan::math::matrix_v;

  matrix_v L(2,2);
  L <<
    1, 0,
    2, 3;
  AVEC x(3);
  x[0] = L(0,0);
  x[1] = L(1,0);
  x[2] = L(1,1);

  matrix_v LLt = tcrossprod(L);
  AVEC y(4);
  y[0] = LLt(0,0);
  y[1] = LLt(0,1);
  y[2] = LLt(1,0);
  y[3] = LLt(1,1);

  EXPECT_FLOAT_EQ(1.0, LLt(0,0).val());
  EXPECT_FLOAT_EQ(2.0, LLt(0,1).val());
  EXPECT_FLOAT_EQ(2.0, LLt(1,0).val());
  EXPECT_FLOAT_EQ(13.0, LLt(1,1).val());

  std::vector<VEC > J;
  stan::math::jacobian(y,x,J);

  // L = 1 0
  //     2 3
  // Jacobian J = Jacobian(L * L')
  // J = 2 0 0
  //     2 1 0
  //     2 1 0
  //     0 4 6
  EXPECT_FLOAT_EQ(2.0,J[0][0]);
  EXPECT_FLOAT_EQ(0.0,J[0][1]);
  EXPECT_FLOAT_EQ(0.0,J[0][2]);

  EXPECT_FLOAT_EQ(2.0,J[1][0]);
  EXPECT_FLOAT_EQ(1.0,J[1][1]);
  EXPECT_FLOAT_EQ(0.0,J[1][2]);

  EXPECT_FLOAT_EQ(2.0,J[2][0]);
  EXPECT_FLOAT_EQ(1.0,J[2][1]);
  EXPECT_FLOAT_EQ(0.0,J[2][2]);

  EXPECT_FLOAT_EQ(0.0,J[3][0]);
  EXPECT_FLOAT_EQ(4.0,J[3][1]);
  EXPECT_FLOAT_EQ(6.0,J[3][2]);
}

TEST(AgradRevMatrix, tcrossprodGrad3) {
  using stan::math::tcrossprod;
  using stan::math::matrix_v;

  matrix_v L(3,3);
  L <<
    1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  AVEC x(6);
  x[0] = L(0,0);
  x[1] = L(1,0);
  x[2] = L(1,1);
  x[3] = L(2,0);
  x[4] = L(2,1);
  x[5] = L(2,2);

  matrix_v LLt = tcrossprod(L);
  AVEC y(9);
  y[0] = LLt(0,0);
  y[1] = LLt(0,1);
  y[2] = LLt(0,2);
  y[3] = LLt(1,0);
  y[4] = LLt(1,1);
  y[5] = LLt(1,2);
  y[6] = LLt(2,0);
  y[7] = LLt(2,1);
  y[8] = LLt(2,2);

  std::vector<VEC > J;
  stan::math::jacobian(y,x,J);

  // L = 1 0 0
  //     2 3 0
  //     4 5 6
  // Jacobian J = Jacobian(L * L')
  // J = 2 0 0 0 0 0
  //     2 1 0 0 0 0
  //     4 0 0 1 0 0
  //     2 1 0 0 0 0
  //     0 4 6 0 0 0
  //     0 4 5 2 3 0
  //     4 0 0 1 0 0
  //     0 4 5 2 3 0
  //     0 0 0 8 10 12

  EXPECT_FLOAT_EQ(2.0,J[0][0]);
  EXPECT_FLOAT_EQ(0.0,J[0][1]);
  EXPECT_FLOAT_EQ(0.0,J[0][2]);
  EXPECT_FLOAT_EQ(0.0,J[0][3]);
  EXPECT_FLOAT_EQ(0.0,J[0][4]);
  EXPECT_FLOAT_EQ(0.0,J[0][5]);

  EXPECT_FLOAT_EQ(2.0,J[1][0]);
  EXPECT_FLOAT_EQ(1.0,J[1][1]);
  EXPECT_FLOAT_EQ(0.0,J[1][2]);
  EXPECT_FLOAT_EQ(0.0,J[1][3]);
  EXPECT_FLOAT_EQ(0.0,J[1][4]);
  EXPECT_FLOAT_EQ(0.0,J[1][5]);

  EXPECT_FLOAT_EQ(4.0,J[2][0]);
  EXPECT_FLOAT_EQ(0.0,J[2][1]);  EXPECT_FLOAT_EQ(0.0,J[2][2]);
  EXPECT_FLOAT_EQ(1.0,J[2][3]);
  EXPECT_FLOAT_EQ(0.0,J[2][4]);
  EXPECT_FLOAT_EQ(0.0,J[2][5]);

  EXPECT_FLOAT_EQ(2.0,J[3][0]);
  EXPECT_FLOAT_EQ(1.0,J[3][1]);
  EXPECT_FLOAT_EQ(0.0,J[3][2]);
  EXPECT_FLOAT_EQ(0.0,J[3][3]);
  EXPECT_FLOAT_EQ(0.0,J[3][4]);
  EXPECT_FLOAT_EQ(0.0,J[3][5]);

  EXPECT_FLOAT_EQ(0.0,J[4][0]);
  EXPECT_FLOAT_EQ(4.0,J[4][1]);
  EXPECT_FLOAT_EQ(6.0,J[4][2]);
  EXPECT_FLOAT_EQ(0.0,J[4][3]);
  EXPECT_FLOAT_EQ(0.0,J[4][4]);
  EXPECT_FLOAT_EQ(0.0,J[4][5]);

  EXPECT_FLOAT_EQ(0.0,J[5][0]);
  EXPECT_FLOAT_EQ(4.0,J[5][1]);
  EXPECT_FLOAT_EQ(5.0,J[5][2]);
  EXPECT_FLOAT_EQ(2.0,J[5][3]);
  EXPECT_FLOAT_EQ(3.0,J[5][4]);
  EXPECT_FLOAT_EQ(0.0,J[5][5]);

  EXPECT_FLOAT_EQ(4.0,J[6][0]);
  EXPECT_FLOAT_EQ(0.0,J[6][1]);
  EXPECT_FLOAT_EQ(0.0,J[6][2]);
  EXPECT_FLOAT_EQ(1.0,J[6][3]);
  EXPECT_FLOAT_EQ(0.0,J[6][4]);
  EXPECT_FLOAT_EQ(0.0,J[6][5]);

  EXPECT_FLOAT_EQ(0.0,J[7][0]);
  EXPECT_FLOAT_EQ(4.0,J[7][1]);
  EXPECT_FLOAT_EQ(5.0,J[7][2]);
  EXPECT_FLOAT_EQ(2.0,J[7][3]);
  EXPECT_FLOAT_EQ(3.0,J[7][4]);
  EXPECT_FLOAT_EQ(0.0,J[7][5]);

  EXPECT_FLOAT_EQ(0.0,J[8][0]);
  EXPECT_FLOAT_EQ(0.0,J[8][1]);
  EXPECT_FLOAT_EQ(0.0,J[8][2]);
  EXPECT_FLOAT_EQ(8.0,J[8][3]);
  EXPECT_FLOAT_EQ(10.0,J[8][4]);
  EXPECT_FLOAT_EQ(12.0,J[8][5]);
}
