#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <complex>
#include <vector>
#include "stan/agrad/agrad.hpp"
#include "stan/agrad/matrix.hpp"
#include <Eigen/Dense>

/*
  MATRIX ADD & SUBTRACT
  mv+mv, mv+md, md+mv
  mv-mv, mv-md, md-mv
  -mv, (-md)
  mv+=mv, mv+=md, (md+=md)
  mv-=mv, mv-=md, (md-=md)

  SCALAR MULTIPLY & DIVIDE
  mv*v, v*mv, d*mv, mv*d, md*v, v*md, (md*d), (d*md)
  mv/v, mv/d, md/v, (md/d)
  mv*=v, mv*=d, (md*=d)
  mv/=v, mv/=d, (md/=d) 

  MATRIX MULTIPLY
  mv*mv, mv*md, md*mv, (md*md)
  mv*=mv, mv*=md, (md*=md)

  TRANSPOSE
  mv.transpose(), [md.transpose()]

  DOT PRODUCT (vectors only)
  mv.dot(mv), [md.dot(md)], !mv.dot(md)!, !md.dot(mv)!

  ARITHMETIC REDUCTIONS
  mv.sum(), [md.sum()]
  mv.prod(), [md.prod()]
  mv.mean(), [md.mean()]
  mv.minCoeff(), [md.minCoeff()]
  mv.maxCoeff(), [md.maxCoeff()]
  mv.trace(), [md.trace()]

  NORMS
  mv.squaredNorm(), [md.squaredNorm()], mv.norm(), [md.norm()]  // squared L2 norms, L2 norms
  mv.lpNorm<1>(), [md.lpNorm<1>()] // L1 norm
  mv.lpNorm<Infinity>(), [md.lpNorm<Infinity>] // L_infinity (max abs value)

  LINEAR ALGEBRA
  mv.determinant(), [md.determinant()] // scalar determinant
  mv.inverse(), [md.inverse()]

  ===========ABOVE HAVE IMPL W. TESTS ===============

  mv.eigenvalues() 
      mv.eigenvalues();  // vector of complex v
      mv.eigenvectors(); // matrix of complex v
  mv.partialPivLu() // LU decomp partial pivot, M=PLU (speed:++, accuracy:+, requires invertibility)
      mv.partialPivLu().matrixLU()   // both L and U matrix in one piece
      mv.partialPivLu().permutationP() // permutation matrix
      mv.partialPivLu().determinant()
  mv.fullPivLu()   // LU decomp full pivot, M = PLUQ (speed:-, accuracy:+++, no reqs)
      mv.partialPivLu().matrixLU()   // both L and U matrix in one piece
      mv.partialPivLu().permutationQ() // permutation matrix P
      mv.partialPivLu().permutationQ() // permutation matrix Q
      mv.fullPivLu().inverse()
      mv.fullPivLu().rank() // returns rank
      mv.fullPivLu().isInvertible() // boolean invertibility
  mv.llt() // Cholesky decomposition, M = LL' (speed:+++, accuracy:+, requires positive definiteness)
      mv.llt().matrixL() // 
      mv.llt().matrixU() // M = UU', too, for upper trinagluar
  SelfAdjointEigenSolver(mv) // Eigenvalue/vector decomposition for symmetric matrices
  EigenSolver(md) // Eigenvalue/vector for arbitrary matrices [NO EigenSolver(mv)]
      md.eigenvalues();  // vector of complex d
      md.eigenvectors();  // vector of complex d
  JacobiSVD(mv) // M = U * diag(S) * V'
      JacobiSVD(mv).singularValues() // S = singular values
      JacobiSVD(mv).matrixU()        // U = singular row vecs
      JacobiSVD(mv).matrixV()        // V = singular col vecs

  PARTIAL REDUCTIONS
  mv.colwise().REDUX(), md.colwise().REDUX() // return row vector of results per column
  mv.rowwise().REDUX(), md.rowwise().REDUX() // return col vector of results per row
  e.g. mv.colwise().mean()

  BROADCASTING
  mv.rowwise() += mv // adds mv to each row
  mv + 1.0; // ?? does this work, saw in Ben's thing

  OTHER UTILITIES
  mv.setRandom() // ?
  mv.setZero();

  FUNCION APPLICATION
  exp(mv), exp(dv), log(mv), log(dv), ...

  BLOCKS, ROWS, COLS
  mv.block(i,j,p,q), md.block(i,j,p,q)  // block size (p,q) start at (i,j)
  mv.row(i), md.row(i)
  mv.col(j), md.col(j)
  mv.head(n), md.head(n)            // first n elts
  mv.tail(n), md.tail(n)            // last n elts
  mv.segment(i,n), md.segment(i,n)  // n elts start at i
 */

using namespace std;
using namespace stan::agrad;
using Eigen::Matrix;
using Eigen::Dynamic;

typedef stan::agrad::var AVAR;
typedef std::vector<AVAR> AVEC;
typedef std::vector<double> VEC;

AVEC createAVEC(AVAR x) {
  AVEC v;
  v.push_back(x);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2) {
  AVEC v = createAVEC(x1);
  v.push_back(x2);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2, AVAR x3) {
  AVEC v = createAVEC(x1,x2);
  v.push_back(x3);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2, AVAR x3, AVAR x4) {
  AVEC v = createAVEC(x1,x2,x3);
  v.push_back(x4);
  return v;
}

VEC cgrad(AVAR f, AVAR x1) {
  AVEC x = createAVEC(x1);
  VEC g;
  f.grad(x,g);
  return g;
}

VEC cgrad(AVAR f, AVAR x1, AVAR x2) {
  AVEC x = createAVEC(x1,x2);
  VEC g;
  f.grad(x,g);
  return g;
}

VEC cgrad(AVAR f, AVAR x1, AVAR x2, AVAR x3) {
  AVEC x = createAVEC(x1,x2,x3);
  VEC g;
  f.grad(x,g);
  return g;
}

VEC cgrad(AVAR f, AVAR x1, AVAR x2, AVAR x3, AVAR x4) {
  AVEC x = createAVEC(x1,x2,x3,x4);
  VEC g;
  f.grad(x,g);
  return g;
}

VEC cgradvec(AVAR f, AVEC x) {
  VEC g;
  f.grad(x,g);
  return g;
}



// all tests pending feedback from Eigen authors

/*

TEST(agrad_matrix,transpose_multiply) {
  mat_double A(2,2);
  A << 1.0, 2.0, 3.0, 4.0;
  EXPECT_EQ(1,1);
  mat_var B(2,2);
  B << -1.0, 0.0, -10.0, 100.0;
  mat_var C = B * A;
}


// vv+vd
TEST(agrad_matrix,add_vv_vd) {
  vec_var A(2);
  A << 1.0, 2.0;
  vec_double B(2);
  B << -1.0, 0.0;

  // AVEC x = createAVEC(A(0,0), A(1,0), A(1,1));
  
  vec_var C = A + B;
}



// mv+mv
TEST(agrad_matrix,add_vv) {
  mat_var a(2,3);
  a << 1.0, 2.0, 3.0,
       4.0, 5.0, 6.0;
  mat_var b(2,3);
  b << 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0;

  AVEC x = createAVEC(a(0,0), a(1,1), b(1,1), b(1,2));

  mat_var c = a + b;
  EXPECT_FLOAT_EQ(2.0,c(0,0).val());
  EXPECT_FLOAT_EQ(2.0,c(0,1).val());
  EXPECT_FLOAT_EQ(4.0,c(0,2).val());
  EXPECT_FLOAT_EQ(4.0,c(1,0).val());
  EXPECT_FLOAT_EQ(6.0,c(1,1).val());
  EXPECT_FLOAT_EQ(6.0,c(1,2).val());
  
  VEC grad = cgradvec(c(1,1),  x);
  EXPECT_FLOAT_EQ(0.0,grad[0]);
  EXPECT_FLOAT_EQ(1.0,grad[1]);
  EXPECT_FLOAT_EQ(1.0,grad[2]);
  EXPECT_FLOAT_EQ(0.0,grad[3]);
}

// mv+md
TEST(agrad_matrix,add_vd) {
  mat_var A(2,3);
  A << 1.0, 2.0, 3.0,
       4.0, 5.0, 6.0;
  mat_double B(2,3);
  B << 1.0, 0.0, 1.0,
    0.0, 1.0, 2.0;

  AVEC x = createAVEC(A(0,0), A(1,0), A(1,1));
  
  mat_var C = A + B;
  EXPECT_FLOAT_EQ(2.0,C(0,0).val());
  EXPECT_FLOAT_EQ(2.0,C(0,1).val());
  EXPECT_FLOAT_EQ(4.0,C(0,2).val());
  EXPECT_FLOAT_EQ(4.0,C(1,0).val());
  EXPECT_FLOAT_EQ(6.0,C(1,1).val());
  EXPECT_FLOAT_EQ(8.0,C(1,2).val());

  VEC grad = cgradvec(C(1,1),  x);
  EXPECT_FLOAT_EQ(0.0,grad[0]);
  EXPECT_FLOAT_EQ(0.0,grad[1]);
  EXPECT_FLOAT_EQ(1.0,grad[2]);
}

// md + mv
TEST(agrad_matrix,add_dv) {
  mat_var A(2,3);
  A << 1.0, 2.0, 3.0,
       4.0, 5.0, 6.0;
  mat_double B(2,3);
  B << 1.0, 0.0, 1.0,
    0.0, 1.0, 2.0;

  AVEC x = createAVEC(A(0,0), A(1,0), A(1,1));
  
  mat_var C = B + A;
  EXPECT_FLOAT_EQ(2.0,C(0,0).val());
  EXPECT_FLOAT_EQ(2.0,C(0,1).val());
  EXPECT_FLOAT_EQ(4.0,C(0,2).val());
  EXPECT_FLOAT_EQ(4.0,C(1,0).val());
  EXPECT_FLOAT_EQ(6.0,C(1,1).val());
  EXPECT_FLOAT_EQ(8.0,C(1,2).val());

  VEC grad = cgradvec(C(1,1),  x);
  EXPECT_FLOAT_EQ(0.0,grad[0]);
  EXPECT_FLOAT_EQ(0.0,grad[1]);
  EXPECT_FLOAT_EQ(1.0,grad[2]);
}

// mv-mv
TEST(agrad_matrix,subtract_vv) {
  mat_var A(2,3);
  A << 1.0, 2.0, 3.0,
       4.0, 5.0, 6.0;
  mat_var B(2,3);
  B << 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0;
  
  mat_var C = A - B;
  EXPECT_FLOAT_EQ(0.0,C(0,0).val());
  EXPECT_FLOAT_EQ(2.0,C(0,1).val());
  EXPECT_FLOAT_EQ(2.0,C(0,2).val());
  EXPECT_FLOAT_EQ(4.0,C(1,0).val());
  EXPECT_FLOAT_EQ(4.0,C(1,1).val());
  EXPECT_FLOAT_EQ(6.0,C(1,2).val());

  VEC grad = cgrad(C(1,1),  A(0,0), A(1,1), B(1,1), B(1,2));
  EXPECT_FLOAT_EQ(0.0,grad[0]);
  EXPECT_FLOAT_EQ(1.0,grad[1]);
  EXPECT_FLOAT_EQ(-1.0,grad[2]);
  EXPECT_FLOAT_EQ(0.0,grad[3]);
}

// mv-md
TEST(agrad_matrix,subtract_vd) {
  mat_var A(2,3);
  A << 1.0, 2.0, 3.0,
       4.0, 5.0, 6.0;
  mat_double B(2,3);
  B << 1.0, 0.0, 1.0,
    0.0, 1.0, 2.0;

  AVEC x = createAVEC(A(0,0), A(1,0), A(1,1));
  
  mat_var C = A - B;
  EXPECT_FLOAT_EQ(0.0,C(0,0).val());
  EXPECT_FLOAT_EQ(2.0,C(0,1).val());
  EXPECT_FLOAT_EQ(2.0,C(0,2).val());
  EXPECT_FLOAT_EQ(4.0,C(1,0).val());
  EXPECT_FLOAT_EQ(4.0,C(1,1).val());
  EXPECT_FLOAT_EQ(4.0,C(1,2).val());

  VEC grad = cgradvec(C(1,1),  x);
  EXPECT_FLOAT_EQ(0.0,grad[0]);
  EXPECT_FLOAT_EQ(0.0,grad[1]);
  EXPECT_FLOAT_EQ(1.0,grad[2]);
}

// md-mv
TEST(agrad_matrix,subtract_dv) {
  mat_double A(2,3);
  A << 1.0, 2.0, 3.0,
       4.0, 5.0, 6.0;
  mat_var B(2,3);
  B << 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0;

  AVEC x = createAVEC(B(0,0), B(1,0), B(1,1));
  
  mat_var C = A - B;
  EXPECT_FLOAT_EQ(0.0,C(0,0).val());
  EXPECT_FLOAT_EQ(2.0,C(0,1).val());
  EXPECT_FLOAT_EQ(2.0,C(0,2).val());
  EXPECT_FLOAT_EQ(4.0,C(1,0).val());
  EXPECT_FLOAT_EQ(4.0,C(1,1).val());
  EXPECT_FLOAT_EQ(6.0,C(1,2).val());

  VEC grad = cgradvec(C(1,1),  x);
  EXPECT_FLOAT_EQ(0.0,grad[0]);
  EXPECT_FLOAT_EQ(0.0,grad[1]);
  EXPECT_FLOAT_EQ(-1.0,grad[2]);
}

// -mv
TEST(agrad_matrix,neg_v) {
  mat_var A(2,3);
  A << 1.0, 2.0, 3.0,
       4.0, 5.0, 6.0;

  AVEC x = createAVEC(A(0,0),A(1,1),A(1,2));
  mat_var C = -A;
  EXPECT_FLOAT_EQ(-1.0,C(0,0).val());
  EXPECT_FLOAT_EQ(-2.0,C(0,1).val());
  EXPECT_FLOAT_EQ(-3.0,C(0,2).val());
  EXPECT_FLOAT_EQ(-4.0,C(1,0).val());
  EXPECT_FLOAT_EQ(-5.0,C(1,1).val());
  EXPECT_FLOAT_EQ(-6.0,C(1,2).val());
  
  VEC g = cgradvec(C(1,1), x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(-1.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}

// mv+=mv
TEST(agrad_matrx, plus_eq_vv) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  mat_var b(2,2);
  b << 10.0, 100.0,
       1000.0, 0.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), b(0,1), b(1,1));

  a += b;
  EXPECT_FLOAT_EQ(11.0,a(0,0).val());
  EXPECT_FLOAT_EQ(102.0,a(0,1).val());
  EXPECT_FLOAT_EQ(1003.0,a(1,0).val());
  EXPECT_FLOAT_EQ(4.0,a(1,1).val());

  VEC g = cgradvec(a(0,1),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(1.0, g[1]);
  EXPECT_FLOAT_EQ(1.0, g[2]);
  EXPECT_FLOAT_EQ(0.0, g[3]);
}

// mv+=md
TEST(agrad_matrx, plus_eq_vd) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  mat_double b(2,2);
  b << 10.0, 100.0,
       1000.0, 0.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1));

  a += b;

  EXPECT_FLOAT_EQ(11.0,a(0,0).val());
  EXPECT_FLOAT_EQ(102.0,a(0,1).val());
  EXPECT_FLOAT_EQ(1003.0,a(1,0).val());
  EXPECT_FLOAT_EQ(4.0,a(1,1).val());

  VEC g = cgradvec(a(0,1),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(1.0, g[1]);
}

// mv-=mv
TEST(agrad_matrx, minus_eq_vv) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  mat_var b(2,2);
  b << 10.0, 100.0,
       1000.0, 0.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), b(0,1), b(1,1));

  a -= b;
  EXPECT_FLOAT_EQ(-9.0,a(0,0).val());
  EXPECT_FLOAT_EQ(-98.0,a(0,1).val());
  EXPECT_FLOAT_EQ(-997.0,a(1,0).val());
  EXPECT_FLOAT_EQ(4.0,a(1,1).val());

  VEC g = cgradvec(a(0,1),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(1.0, g[1]);
  EXPECT_FLOAT_EQ(-1.0, g[2]);
  EXPECT_FLOAT_EQ(0.0, g[3]);
}

// mv-=md
TEST(agrad_matrx, minus_eq_vd) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  mat_double b(2,2);
  b << 10.0, 100.0,
       1000.0, 0.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1));

  a -= b;

  EXPECT_FLOAT_EQ(-9.0,a(0,0).val());
  EXPECT_FLOAT_EQ(-98.0,a(0,1).val());
  EXPECT_FLOAT_EQ(-997.0,a(1,0).val());
  EXPECT_FLOAT_EQ(4.0,a(1,1).val());

  VEC g = cgradvec(a(0,1),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(1.0, g[1]);
}

// mv*v
TEST(agrad_matrx, times_mv_v) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  AVAR b = 7.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), b);

  mat_var c = a * b;
  EXPECT_FLOAT_EQ(7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(14.0,c(0,1).val());
  EXPECT_FLOAT_EQ(21.0,c(1,0).val());
  EXPECT_FLOAT_EQ(28.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(7.0, g[2]);
  EXPECT_FLOAT_EQ(3.0, g[3]);
}

// v*mv
TEST(agrad_matrx, times_v_mv) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  AVAR b = 7.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), b);

  mat_var c = b * a;
  EXPECT_FLOAT_EQ(7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(14.0,c(0,1).val());
  EXPECT_FLOAT_EQ(21.0,c(1,0).val());
  EXPECT_FLOAT_EQ(28.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(7.0, g[2]);
  EXPECT_FLOAT_EQ(3.0, g[3]);
}

// mv*d
TEST(agrad_matrx, times_mv_d) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  double b = 7.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0));

  mat_var c = a * b;
  EXPECT_FLOAT_EQ(7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(14.0,c(0,1).val());
  EXPECT_FLOAT_EQ(21.0,c(1,0).val());
  EXPECT_FLOAT_EQ(28.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(7.0, g[2]);
}

// d*mv
TEST(agrad_matrx, times_d_mv) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  double b = 7.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0));

  mat_var c = b * a;
  EXPECT_FLOAT_EQ(7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(14.0,c(0,1).val());
  EXPECT_FLOAT_EQ(21.0,c(1,0).val());
  EXPECT_FLOAT_EQ(28.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(7.0, g[2]);
}

// md*v
TEST(agrad_matrx, times_md_v) {
  mat_double a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  AVAR b = 7.0;
  
  AVEC x = createAVEC(b);

  mat_var c = a * b;
  EXPECT_FLOAT_EQ(7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(14.0,c(0,1).val());
  EXPECT_FLOAT_EQ(21.0,c(1,0).val());
  EXPECT_FLOAT_EQ(28.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(3.0, g[0]);
}

// v*md
TEST(agrad_matrx, times_v_md) {
  mat_double a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  AVAR b = 7.0;
  
  AVEC x = createAVEC(b);

  mat_var c = b * a;
  EXPECT_FLOAT_EQ(7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(14.0,c(0,1).val());
  EXPECT_FLOAT_EQ(21.0,c(1,0).val());
  EXPECT_FLOAT_EQ(28.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(3.0, g[0]);
}



// mv/v
TEST(agrad_matrx, div_mv_v) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  AVAR b = 7.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), b);

  mat_var c = a / b;
  EXPECT_FLOAT_EQ(1.0/7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(2.0/7.0,c(0,1).val());
  EXPECT_FLOAT_EQ(3.0/7.0,c(1,0).val());
  EXPECT_FLOAT_EQ(4.0/7.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(1.0/7.0, g[2]);
  EXPECT_FLOAT_EQ(-3.0/(7.0*7.0), g[3]);
}

// mv/d
TEST(agrad_matrx, div_mv_d) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  double b = 7.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0));

  mat_var c = a / b;
  EXPECT_FLOAT_EQ(1.0/7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(2.0/7.0,c(0,1).val());
  EXPECT_FLOAT_EQ(3.0/7.0,c(1,0).val());
  EXPECT_FLOAT_EQ(4.0/7.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(1.0/7.0, g[2]);
}

// md/v
TEST(agrad_matrx, div_md_v) {
  mat_double a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  AVAR b = 7.0;
  
  AVEC x = createAVEC(b);

  mat_var c = a / b;
  EXPECT_FLOAT_EQ(1.0/7.0,c(0,0).val());
  EXPECT_FLOAT_EQ(2.0/7.0,c(0,1).val());
  EXPECT_FLOAT_EQ(3.0/7.0,c(1,0).val());
  EXPECT_FLOAT_EQ(4.0/7.0,c(1,1).val());

  VEC g = cgradvec(c(1,0),x);
  EXPECT_FLOAT_EQ(-3.0/(7.0*7.0), g[0]);
}

// mv*=v
TEST(agrad_matrx, mv_times_eq_v) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  AVAR b = 7.0;

  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), b);

  a *= b;

  EXPECT_FLOAT_EQ(7.0,a(0,0).val());
  EXPECT_FLOAT_EQ(14.0,a(0,1).val());
  EXPECT_FLOAT_EQ(21.0,a(1,0).val());
  EXPECT_FLOAT_EQ(28.0,a(1,1).val());

  VEC g = cgradvec(a(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(7.0, g[2]);
  EXPECT_FLOAT_EQ(3.0, g[3]);
}

// mv*=d
TEST(agrad_matrx, mv_times_eq_d) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  double b = 7.0;

  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0));

  a *= b;

  EXPECT_FLOAT_EQ(7.0,a(0,0).val());
  EXPECT_FLOAT_EQ(14.0,a(0,1).val());
  EXPECT_FLOAT_EQ(21.0,a(1,0).val());
  EXPECT_FLOAT_EQ(28.0,a(1,1).val());

  VEC g = cgradvec(a(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(7.0, g[2]);
}

// mv/=v
TEST(agrad_matrx, mv_div_eq_v) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  AVAR b = 7.0;

  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), b);

  a /= b;

  EXPECT_FLOAT_EQ(1.0/7.0,a(0,0).val());
  EXPECT_FLOAT_EQ(2.0/7.0,a(0,1).val());
  EXPECT_FLOAT_EQ(3.0/7.0,a(1,0).val());
  EXPECT_FLOAT_EQ(4.0/7.0,a(1,1).val());

  VEC g = cgradvec(a(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(1.0/7.0, g[2]);
  EXPECT_FLOAT_EQ(-3.0/(7.0 * 7.0), g[3]);
}

// mv/=d
TEST(agrad_matrx, mv_div_eq_d) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;
  
  double b = 7.0;

  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0));

  a /= b;

  EXPECT_FLOAT_EQ(1.0/7.0,a(0,0).val());
  EXPECT_FLOAT_EQ(2.0/7.0,a(0,1).val());
  EXPECT_FLOAT_EQ(3.0/7.0,a(1,0).val());
  EXPECT_FLOAT_EQ(4.0/7.0,a(1,1).val());

  VEC g = cgradvec(a(1,0),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(1.0/7.0, g[2]);
}


// mv*mv
TEST(agrad_matrx, mv_times_mv) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;

  mat_var b(2,3);
  b << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;;

  AVEC x = createAVEC(a(0,0), a(1,0), b(1,0), b(0,2));
  x.push_back(b(1,2));

  mat_var c = a * b;

  EXPECT_FLOAT_EQ(9.0, c(0,0).val());
  EXPECT_FLOAT_EQ(22.0, c(0,1).val());
  EXPECT_FLOAT_EQ(197.0, c(0,2).val());
  EXPECT_FLOAT_EQ(17.0, c(1,0).val());
  EXPECT_FLOAT_EQ(46.0, c(1,1).val());
  EXPECT_FLOAT_EQ(391.0, c(1,2).val());

  VEC g = cgradvec(c(1,2),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(-3.0, g[1]);
  EXPECT_FLOAT_EQ(0.0, g[2]);
  EXPECT_FLOAT_EQ(3.0, g[3]);
  EXPECT_FLOAT_EQ(4.0, g[4]);
}

// mv*md
TEST(agrad_matrx, mv_times_md) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;

  mat_double b(2,3);
  b << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;;

  AVEC x = createAVEC(a(0,0), a(1,0));

  mat_var c = a * b;

  EXPECT_FLOAT_EQ(9.0, c(0,0).val());
  EXPECT_FLOAT_EQ(22.0, c(0,1).val());
  EXPECT_FLOAT_EQ(197.0, c(0,2).val());
  EXPECT_FLOAT_EQ(17.0, c(1,0).val());
  EXPECT_FLOAT_EQ(46.0, c(1,1).val());
  EXPECT_FLOAT_EQ(391.0, c(1,2).val());

  VEC g = cgradvec(c(1,2),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(-3.0, g[1]);
}

// md*mv
TEST(agrad_matrx, md_times_mv) {
  mat_double a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;

  mat_var b(2,3);
  b << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;;

  AVEC x = createAVEC(b(1,0), b(0,2), b(1,2));

  mat_var c = a * b;

  EXPECT_FLOAT_EQ(9.0, c(0,0).val());
  EXPECT_FLOAT_EQ(22.0, c(0,1).val());
  EXPECT_FLOAT_EQ(197.0, c(0,2).val());
  EXPECT_FLOAT_EQ(17.0, c(1,0).val());
  EXPECT_FLOAT_EQ(46.0, c(1,1).val());
  EXPECT_FLOAT_EQ(391.0, c(1,2).val());

  VEC g = cgradvec(c(1,2),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(3.0, g[1]);
  EXPECT_FLOAT_EQ(4.0, g[2]);
}

// mv*=mv
TEST(agrad_matrx, mv_times_eq_mv) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;

  mat_var b(2,3);
  b << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;;

  AVEC x = createAVEC(a(0,0), a(1,0), b(1,0), b(0,2));
  x.push_back(b(1,2));

  a *= b;

  EXPECT_FLOAT_EQ(9.0, a(0,0).val());
  EXPECT_FLOAT_EQ(22.0, a(0,1).val());
  EXPECT_FLOAT_EQ(197.0, a(0,2).val());
  EXPECT_FLOAT_EQ(17.0, a(1,0).val());
  EXPECT_FLOAT_EQ(46.0, a(1,1).val());
  EXPECT_FLOAT_EQ(391.0, a(1,2).val());

  VEC g = cgradvec(a(1,2),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(-3.0, g[1]);
  EXPECT_FLOAT_EQ(0.0, g[2]);
  EXPECT_FLOAT_EQ(3.0, g[3]);
  EXPECT_FLOAT_EQ(4.0, g[4]);
}


// mv*=md
TEST(agrad_matrx, mv_times_eq_md) {
  mat_var a(2,2);
  a << 1.0, 2.0,
       3.0, 4.0;

  mat_double b(2,3);
  b << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;;

  AVEC x = createAVEC(a(0,0), a(1,0));

  a *= b;

  EXPECT_FLOAT_EQ(9.0, a(0,0).val());
  EXPECT_FLOAT_EQ(22.0, a(0,1).val());
  EXPECT_FLOAT_EQ(197.0, a(0,2).val());
  EXPECT_FLOAT_EQ(17.0, a(1,0).val());
  EXPECT_FLOAT_EQ(46.0, a(1,1).val());
  EXPECT_FLOAT_EQ(391.0, a(1,2).val());

  VEC g = cgradvec(a(1,2),x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(-3.0, g[1]);
}

// mv.transpose()
TEST(agrad_matrix,transpose) {
  mat_var a(2,3);
  a << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;
  
  AVEC x = createAVEC(a(0,0), a(0,2), a(1,1));
  
  mat_var c = a.transpose();
  EXPECT_FLOAT_EQ(-1.0,c(0,0).val());
  EXPECT_FLOAT_EQ(10.0,c(1,1).val());
  EXPECT_FLOAT_EQ(-3.0,c(2,0).val());
  EXPECT_EQ(3,c.rows());
  EXPECT_EQ(2,c.cols());

  VEC g = cgradvec(c(2,0),x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}

TEST(agrad_matrix,vector_vector_multiply) {
  mat_var a(2,1);
  AVAR a00 = 2.0;
  AVAR a01 = 3.0;
  a << a00, a01;

  mat_var b(1,2);
  AVAR b00 = 5.0;
  AVAR b10 = 7.0;
  b << b00, b10;
  
  mat_var Z = b * a;
  EXPECT_EQ(1,Z.rows());
  EXPECT_EQ(1,Z.cols());
  AVAR z_0_0 = Z(0,0);
  EXPECT_FLOAT_EQ(31.0, z_0_0.val());

  VEC g = cgrad(z_0_0,  a00, a01, b00, b10);
  EXPECT_FLOAT_EQ(5.0, g[0]);
  EXPECT_FLOAT_EQ(7.0, g[1]);
  EXPECT_FLOAT_EQ(2.0, g[2]);
  EXPECT_FLOAT_EQ(3.0, g[3]);
}


// mv.dot(mv)
TEST(agrad_matrix,mv_dot_mv) {
  vec_var a(2);
  a << 2.0, 3.0;

  vec_var b(2);
  b << 5.0, 7.0;
  
  AVEC x = createAVEC(a(0,0), a(1,0), b(0,0), b(1,0));

  AVAR z = a.dot(b);
  EXPECT_FLOAT_EQ(31.0, z.val());
  
  VEC g = cgradvec(z,x);
  EXPECT_FLOAT_EQ(5.0, g[0]);
  EXPECT_FLOAT_EQ(7.0, g[1]);
  EXPECT_FLOAT_EQ(2.0, g[2]);
  EXPECT_FLOAT_EQ(3.0, g[3]);
}

// mv.dot(dv): COMPILER ERROR

// dv.dot(mv): COMPILER ERROR

// mv.sum()
TEST(agrad_matrix,mv_sum) {
  mat_var a(2,3);
  a << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;
  
  AVEC x = createAVEC(a(0,0), a(0,2), a(1,1));

  AVAR s = a.sum();
  EXPECT_FLOAT_EQ(113.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(1.0, g[0]);
  EXPECT_FLOAT_EQ(1.0, g[1]);
  EXPECT_FLOAT_EQ(1.0, g[2]);
}  

// mv.prod()
TEST(agrad_matrix,mv_prod) {
  mat_var a(2,2);
  a << 1.0, 2.0, 
       3.0, 4.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = a.prod();
  EXPECT_FLOAT_EQ(24.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(24.0, g[0]);
  EXPECT_FLOAT_EQ(12.0, g[1]);
  EXPECT_FLOAT_EQ(8.0, g[2]);
  EXPECT_FLOAT_EQ(6.0, g[3]);
}

// mv.mean()
TEST(agrad_matrix,mv_mean) {
  mat_var a(2,3);
  a << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;
  
  AVEC x = createAVEC(a(0,0), a(0,2), a(1,1));

  AVAR s = a.mean();
  EXPECT_FLOAT_EQ(113.0/6.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(1.0/6.0, g[0]);
  EXPECT_FLOAT_EQ(1.0/6.0, g[1]);
  EXPECT_FLOAT_EQ(1.0/6.0, g[2]);
}  

// mv.minCoeff()
TEST(agrad_matrix,mv_min) {
  mat_var a(2,3);
  a << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;
  
  AVEC x = createAVEC(a(0,0), a(0,2), a(1,1));

  AVAR s = a.minCoeff();
  EXPECT_FLOAT_EQ(-3.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(1.0, g[1]);
  EXPECT_FLOAT_EQ(0.0, g[2]);
}  

// mv.maxCoeff()
TEST(agrad_matrix,mv_max) {
  mat_var a(2,3);
  a << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;
  
  AVEC x = createAVEC(a(0,0), a(0,2), a(1,1), a(1,2));

  AVAR s = a.maxCoeff();
  EXPECT_FLOAT_EQ(100.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(0.0, g[2]);
  EXPECT_FLOAT_EQ(1.0, g[3]);
}  


// mv.trace()
TEST(agrad_matrix,mv_trace) {
  mat_var a(2,2);
  a << -1.0, 2.0, 
       5.0, 10.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = a.trace();
  EXPECT_FLOAT_EQ(9.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(1.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(0.0, g[2]);
  EXPECT_FLOAT_EQ(1.0, g[3]);
}  

TEST(agrad_matrix,mv_squaredNorm) {
  mat_var a(2,2);
  a << -1.0, 2.0, 
       5.0, 10.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = a.squaredNorm();
  EXPECT_FLOAT_EQ(130.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(-2.0, g[0]);
  EXPECT_FLOAT_EQ(4.0, g[1]);
  EXPECT_FLOAT_EQ(10.0, g[2]);
  EXPECT_FLOAT_EQ(20.0, g[3]);
}  

TEST(agrad_matrix,mv_norm) {
  mat_var a(2,1);
  a << -3.0, 4.0;
  
  AVEC x = createAVEC(a(0,0), a(1,0));

  AVAR s = a.norm();
  EXPECT_FLOAT_EQ(5.0,s.val());

  // (see hypot in special_functions_test) 
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(-3.0/5.0, g[0]);
  EXPECT_FLOAT_EQ(4.0/5.0, g[1]);
}  

TEST(agrad_matrix,mv_lp_norm) {
  mat_var a(2,2);
  a << -1.0, 2.0, 
    5.0, 0.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = a.lpNorm<1>();
  EXPECT_FLOAT_EQ(8.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(-1.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(1.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]); // ? depends on impl here, could be -1 or 1
}  

TEST(agrad_matrix,mv_lp_norm_inf) {
  mat_var a(2,2);
  a << -1.0, 2.0, 
    -5.0, 0.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = a.lpNorm<Eigen::Infinity>();
  EXPECT_FLOAT_EQ(5.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(-1.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]); 
}  

TEST(agrad_matrix,det) {
  mat_var X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, c, d;

  AVAR f = X.determinant();

  // det = ad - bc
  EXPECT_FLOAT_EQ(-1.0,f.val());

  AVEC x = createAVEC(a,b,c,d);
  std::vector<double> g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(7.0,g[0]);
  EXPECT_FLOAT_EQ(-5.0,g[1]);
  EXPECT_FLOAT_EQ(-3.0,g[2]);
  EXPECT_FLOAT_EQ(2.0,g[3]);

   // just test it can handle it
  mat_var Z(9,9);
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      Z(i,j) = i * j + 1;
  AVAR h = Z.determinant();
}

TEST(agrad_matrix,inverse) {
  mat_var a(2,2);
  a << 2.0, 3.0, 
       5.0, 7.0;
  AVEC x = createAVEC(a(0,0),a(0,1),a(1,0),a(1,1));

  mat_var a_inv = a.inverse();

  mat_double ad(2,2);
  ad << 2.0, 3.0, 
       5.0, 7.0;
  mat_double ad_inv = ad.inverse();

  int k = 0;
  int l = 1;
  VEC g = cgradvec(a_inv(k,l),x);

  int idx = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_FLOAT_EQ(-ad_inv(k,i) * ad_inv(j,l), g[idx]);
      ++idx;
    }
  }

  mat_var I = a * a_inv;

  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  mat_var I2 = a_inv * a;
  EXPECT_NEAR(1.0,I2(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I2(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I2(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I2(1,1).val(),1.0e-12);
}

TEST(agrad_matrix,eigenval_sum) {
  mat_var a(3,3);
  a << 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 13.0, 11.0, 19.0;
  AVEC x = createAVEC(a(0,0), a(1,1), a(2,2), a(1,2));
  x.push_back(a(0,1));
  x.push_back(a(2,0));

  // grad sum eig = I
  AVAR a_eigenvalues_sum = a.eigenvalues().real().sum();
  
  VEC g = cgradvec(a_eigenvalues_sum,x);
  EXPECT_NEAR(1.0,g[0],1.0E-11);
  EXPECT_NEAR(1.0,g[1],1.0E-11);
  EXPECT_NEAR(1.0,g[2],1.0E-11);

  EXPECT_NEAR(0.0,g[3],1.0E-10);
  EXPECT_NEAR(0.0,g[4],1.0E-10);
  EXPECT_NEAR(0.0,g[5],1.0E-10);
}


// COMPOUND TESTS BELOW

TEST(agrad_matrix,matrix_matrix_multiply_add) {
  mat_var X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, c, d;

  EXPECT_FLOAT_EQ(2.0,X(0,0).val());
  EXPECT_FLOAT_EQ(3.0,X(0,1).val());
  EXPECT_FLOAT_EQ(5.0,X(1,0).val());
  EXPECT_FLOAT_EQ(7.0,X(1,1).val());

  mat_var Y(2,2);
  AVAR e = 11.0;
  AVAR f = 13.0;
  AVAR g = 17.0;
  AVAR h = 19.0;
  Y << e, f, g, h;

  mat_var Z = X * Y;

  EXPECT_FLOAT_EQ(73.0,Z(0,0).val());
  EXPECT_FLOAT_EQ(83.0,Z(0,1).val());
  EXPECT_FLOAT_EQ(174.0,Z(1,0).val());
  EXPECT_FLOAT_EQ(198.0,Z(1,1).val());

  AVAR result = Z(0,0) + Z(0,1) + Z(1,0) + Z(1,1);

  EXPECT_FLOAT_EQ(528.0,result.val());
  AVEC x = createAVEC(a,b,c,d);
  x.push_back(e);
  x.push_back(f);
  x.push_back(g);
  x.push_back(h);
  std::vector<double> grad;
  result.grad(x,grad);

  EXPECT_FLOAT_EQ(24.0,grad[0]); // e + f
  EXPECT_FLOAT_EQ(36.0,grad[1]); // g + h
  EXPECT_FLOAT_EQ(24.0,grad[2]); // e + f
  EXPECT_FLOAT_EQ(36.0,grad[3]); // g + h
  EXPECT_FLOAT_EQ(7.0,grad[4]); // a + c
  EXPECT_FLOAT_EQ(7.0,grad[5]); // a + c
  EXPECT_FLOAT_EQ(10.0,grad[6]); // b + d
  EXPECT_FLOAT_EQ(10.0,grad[7]); // b + d
}

TEST(agrad_matrix,inverse_inverse_sum) {
  mat_var a(4,4);
  a << 2.0, 3.0, 4.0, 5.0, 
    9.0, -1.0, 2.0, 2.0,
    4.0, 3.0, 7.0, -1.0,
    0.0, 1.0, 19.0, 112.0;

  AVEC x;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      x.push_back(a(i,j));

  AVAR a_inv_inv_sum = a.inverse().inverse().sum();

  VEC g = cgradvec(a_inv_inv_sum,x);

  for (unsigned int k = 0; k < x.size(); ++k)
    EXPECT_FLOAT_EQ(1.0,g[k]);
}



TEST(agrad_matrix,eigenvalues) {
  mat_var X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, c, d;

  // general eigenvalues case is complex
  mat_var X_eig = X.eigenvalues().real();

  // compiles, but seg faults on AVAR
  // Eigen::EigenSolver<mat_var > solver(X);

  Matrix<double,Dynamic,Dynamic> Y(2,2);
  Y << 2.0, 3.0, 3.0, 7.0;
  Eigen::EigenSolver<Matrix<double,Dynamic,Dynamic> > solver(Y);
}

TEST(agrad_matrix,eigenvalues_self_adj) {
  mat_var X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, 
       c, d;

  Eigen::SelfAdjointEigenSolver<mat_var > solver(X);
  vec_var eigenvalues = 
    solver.eigenvalues();
  mat_var eigenvectors = 
    solver.eigenvectors();

}

TEST(agrad_matrix,scalar_var_products) {
  mat_var X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, c, d;

  mat_var Z = 5.0 * X;
  EXPECT_FLOAT_EQ(10.0,Z(0,0).val());
  EXPECT_FLOAT_EQ(15.0,Z(0,1).val());
  EXPECT_FLOAT_EQ(25.0,Z(1,0).val());
  EXPECT_FLOAT_EQ(35.0,Z(1,1).val());
}

TEST(agrad_matrix,exp) {
  mat_var X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, c, d;

  // REQS DEFN ABOVE TO BE INCLUDED
  mat_var Z = exp(X);
  EXPECT_FLOAT_EQ(exp(2.0),Z(0,0).val());
  EXPECT_FLOAT_EQ(exp(3.0),Z(0,1).val());
  EXPECT_FLOAT_EQ(exp(5.0),Z(1,0).val());
  EXPECT_FLOAT_EQ(exp(7.0),Z(1,1).val());
}


TEST(agrad_matrix,vec_mult_two_ways) {
  mat_var X(2,1);
  AVAR a = 2.0;
  AVAR b = 3.0;
  X << a, b;

  mat_var Y(1,2);
  AVAR c = 5.0;
  AVAR d = 7.0;
  Y << c, d;
  
  mat_var Z_22 = X * Y;
  mat_var Z_11 = Y * X;
}

TEST(agrad_matrix,vec_mult_diff_types) {
  mat_var X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b,
       c, d;
  
  mat_var X2 = X;

  Matrix<double,Dynamic,Dynamic> Y(2,2);
  double e = 9.0;
  double f = 11.0;
  double g = 13.0;
  double h = 17.0;
  
  Matrix<double,Dynamic,Dynamic> Y2 = Y;

  Y << e, f,
       g, h;

  mat_var Z_times_vd = X * Y;
  mat_var Z_times_dv = Y * X;

  mat_var Z_plus_vd = X + Y;
  mat_var Z_plus_dv = Y + X;

  mat_var Z_minus_vd = X - Y;
  mat_var Z_minus_dv = Y - X;

  mat_var Z_minus_v = -X;
  Matrix<double,Dynamic,Dynamic> Z_minus_d = -Y;

  // Mixed doesn't work w/o conversion
  // X += Y;
  X += X;
  Y += Y;
  
  X -= X2;
  Y -= Y2;

  Y *= 2.0;
  var z(2.0);
  X *= z;
  
}

TEST(agrad_matrix,mat_cholesky) {
  // symmetric
  mat_var X(2,2);
  AVAR a = 3.0;
  AVAR b = -1.0;
  AVAR c = -1.0;
  AVAR d = 1.0;
  X << a, b, 
       c, d;
  
  Eigen::LLT<mat_var > llt = X.llt();
  mat_var L = llt.matrixL();
  mat_var U = llt.matrixU();

  mat_var LU = L * U;
  EXPECT_FLOAT_EQ(a.val(),LU(0,0).val());
  EXPECT_FLOAT_EQ(b.val(),LU(0,1).val());
  EXPECT_FLOAT_EQ(c.val(),LU(1,0).val());
  EXPECT_FLOAT_EQ(d.val(),LU(1,1).val());
}


TEST(agrad_matrix,mat_lu) {
  mat_var X(2,2);
  AVAR a = 3.0;
  AVAR b = -1.0;
  AVAR c = -2.0;
  AVAR d = 5.0;
  X << a, b, 
       c, d;

  Eigen::PartialPivLU<mat_var > lu = X.lu();
  // L is lower and U upper part of matrix!
  mat_var LU= lu.matrixLU();
  Eigen::PermutationMatrix<Dynamic> P = lu.permutationP();
  
  AVAR det = lu.determinant();
  // det = ad - bc
  EXPECT_NEAR(3.0 * 5.0 - (-1.0 * -2.0), det.val(), 1E-14);

  mat_var X_inv = X.inverse();

  mat_var I = X * X_inv;
  EXPECT_FLOAT_EQ(1.0,I(0,0).val());
  EXPECT_NEAR(0.0,I(0,1).val(),1E-14);
  EXPECT_NEAR(0.0,I(1,0).val(),1E-14);
  EXPECT_FLOAT_EQ(1.0,I(0,0).val());



}

*/




