#include <stan/math/matrix/promoter.hpp>
#include <stan/math/matrix/promote_common.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>

TEST(AgradRevMatrix, promoter) {
  using stan::math::promoter;
  using stan::math::promote_common;

  using stan::agrad::var;

  using std::vector;

  using stan::math::matrix_d;
  using stan::math::vector_d;
  using stan::math::row_vector_d;

  using stan::agrad::matrix_v;
  using stan::agrad::vector_v;
  using stan::agrad::row_vector_v;
  
  double a = promote_common<double,double>(3.0);
  var b = promote_common<double,var>(4.0);
  var c = promote_common<double,var>(var(5.0));
  EXPECT_FLOAT_EQ(3.0, a);
  EXPECT_FLOAT_EQ(4.0, b.val());
  EXPECT_FLOAT_EQ(5.0, c.val());

  vector<double> x(5);
  for (int i = 0; i < 5; ++i) x[i] = i * i;
  vector<double> y = promote_common<vector<double>, vector<double> >(x);
  EXPECT_EQ(5U,y.size());
  for (int i = 0; i < 5; ++i)
    EXPECT_FLOAT_EQ(x[i],y[i]);
  std::vector<var> z = 
    promote_common<std::vector<double>, std::vector<var> >(x);
  EXPECT_EQ(5U,z.size());
  for (int i = 0; i < 5; ++i)
    EXPECT_FLOAT_EQ(x[i],z[i].val());
  vector<var> w = promote_common<vector<var>, vector<var> >(z);
  EXPECT_EQ(5U,w.size());
  for (int i = 0; i < 5; ++i)
    EXPECT_FLOAT_EQ(x[i],w[i].val());
  
  vector_d U(3);
  for (int i = 0; i < 3; ++i)
    U(i) = i * i;
  vector_d V = promote_common<vector_d,vector_d>(U);
  EXPECT_EQ(3,V.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(U(i), V(i));
  vector_v W = promote_common<vector_d,vector_v>(U);
  EXPECT_EQ(3,W.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(U(i), W(i).val());
  vector_v X = promote_common<vector_v,vector_v>(W);
  EXPECT_EQ(3,X.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(U(i), X(i).val());


  row_vector_d G(3);
  for (int i = 0; i < 3; ++i)
    G(i) = i * i;
  row_vector_d H = promote_common<row_vector_d,row_vector_d>(G);
  EXPECT_EQ(3,H.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(G(i), H(i));
  row_vector_v I = promote_common<row_vector_d,row_vector_v>(G);
  EXPECT_EQ(3,W.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(G(i), I(i).val());
  row_vector_v J = promote_common<row_vector_v,row_vector_v>(I);
  EXPECT_EQ(3,J.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(H(i), J(i).val());

  matrix_d A(3,4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      A(i,j) = (i + 1) + (j * 10);
  matrix_d B = promote_common<matrix_d,matrix_d>(A);
  EXPECT_EQ(3,B.rows());
  EXPECT_EQ(4,B.cols());
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(A(i,j),B(i,j));
  matrix_v C = promote_common<matrix_d,matrix_v>(A);
  EXPECT_EQ(3,C.rows());
  EXPECT_EQ(4,C.cols());
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(A(i,j),C(i,j).val());
  matrix_v D = promote_common<matrix_v,matrix_v>(C);
  EXPECT_EQ(3,D.rows());
  EXPECT_EQ(4,D.cols());
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(A(i,j),D(i,j).val());
}
