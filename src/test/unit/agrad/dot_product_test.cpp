#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix.hpp>
#include <stan/agrad/rev/matrix.hpp>

// this didn't need to be external, so it's here now:


TEST(AgradRevMatrix, dot_product_vv) {
  AVEC a, b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  AVEC ab;
  VEC grad;
  for (size_t i = 0; i < 3; i++) {
    ab.push_back(a[i]);
    ab.push_back(b[i]);
  }
  c.grad(ab, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], -1);
  EXPECT_EQ(grad[2], 2);
  EXPECT_EQ(grad[3], 0);
  EXPECT_EQ(grad[4], 3);
  EXPECT_EQ(grad[5], 1);
}
TEST(AgradRevMatrix, dot_product_dv) {
  VEC a;
  AVEC b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  VEC grad;
  c.grad(b, grad);
  EXPECT_EQ(grad[0], -1);
  EXPECT_EQ(grad[1], 0);
  EXPECT_EQ(grad[2], 1);
}
TEST(AgradRevMatrix, dot_product_vd) {
  AVEC a;
  VEC b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  VEC grad;
  c.grad(a, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], 2);
  EXPECT_EQ(grad[2], 3);
}
TEST(AgradRevMatrix, dot_product_vv_vec) {
  AVEC a, b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  AVEC ab;
  VEC grad;
  for (size_t i = 0; i < 3; i++) {
    ab.push_back(a[i]);
    ab.push_back(b[i]);
  }
  c.grad(ab, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], -1);
  EXPECT_EQ(grad[2], 2);
  EXPECT_EQ(grad[3], 0);
  EXPECT_EQ(grad[4], 3);
  EXPECT_EQ(grad[5], 1);
}
TEST(AgradRevMatrix, dot_product_dv_vec) {
  VEC a;
  AVEC b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  VEC grad;
  c.grad(b, grad);
  EXPECT_EQ(grad[0], -1);
  EXPECT_EQ(grad[1], 0);
  EXPECT_EQ(grad[2], 1);
}
TEST(AgradRevMatrix, dot_product_vd_vec) {
  AVEC a;
  VEC b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  VEC grad;
  c.grad(a, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], 2);
  EXPECT_EQ(grad[2], 3);
}

template <int R, int C>
void assert_val_grad(Eigen::Matrix<stan::agrad::var,R,C>& v) {
  v << -1.0, 0.0, 3.0;
  AVEC x = createAVEC(v(0),v(1),v(2));
  AVAR f = dot_self(v);
  VEC g;
  f.grad(x,g);
  
  EXPECT_FLOAT_EQ(-2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(6.0,g[2]);
}  


TEST(AgradRevMatrix, dot_self_vec) {
  using stan::math::dot_self;

  Eigen::Matrix<AVAR,Eigen::Dynamic,1> v1(1);
  v1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(v1).val(),1E-12);
  Eigen::Matrix<AVAR,Eigen::Dynamic,1> v2(2);
  v2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(v2).val(),1E-12);
  Eigen::Matrix<AVAR,Eigen::Dynamic,1> v3(3);
  v3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(v3).val(),1E-12);  

  Eigen::Matrix<AVAR,Eigen::Dynamic,1> v(3);
  assert_val_grad(v);

  Eigen::Matrix<AVAR,1,Eigen::Dynamic> vv(3);
  assert_val_grad(vv);

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> vvv(3,1);
  assert_val_grad(vvv);

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> vvvv(1,3);
  assert_val_grad(vvvv);
}

TEST(AgradRevMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val(),1E-12);
  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val(),1E-12);
  EXPECT_NEAR(9.0,x(0,1).val(),1E-12);
  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val(),1E-12);
  EXPECT_NEAR(34.0,x(0,1).val(),1E-12);

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> vvv(3,1);
  assert_val_grad(vvv);

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> vvvv(1,3);
  assert_val_grad(vvvv);
}

TEST(AgradRevMatrix,softmax) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::vector_v;

  EXPECT_THROW(softmax(vector_v()),std::domain_error);
  
  Matrix<AVAR,Dynamic,1> x(1);
  x << 0.0;
  
  Matrix<AVAR,Dynamic,1> theta = softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0].val());

  Matrix<AVAR,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
  Matrix<AVAR,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0].val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1].val());

  Matrix<AVAR,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
  Matrix<AVAR,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0].val());
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1].val());
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2].val());
}
TEST(AgradRevMatrix, meanStdVector) {
  using stan::math::mean; // should use arg-dep lookup
  AVEC x(0);
  EXPECT_THROW(mean(x), std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0, mean(x).val());
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(1.5, mean(x).val());

  AVEC y = createAVEC(1.0,2.0);
  AVAR f = mean(y);
  VEC grad = cgrad(f, y[0], y[1]);
  EXPECT_FLOAT_EQ(0.5, grad[0]);
  EXPECT_FLOAT_EQ(0.5, grad[1]);
  EXPECT_EQ(2U, grad.size());
}
TEST(AgradRevMatrix, varianceStdVector) {
  using stan::math::variance; // should use arg-dep lookup

  AVEC y1 = createAVEC(0.5,2.0,3.5);
  AVAR f1 = variance(y1);
  VEC grad1 = cgrad(f1, y1[0], y1[1], y1[2]);
  double f1_val = f1.val(); // save before cleaned out

  AVEC y2 = createAVEC(0.5,2.0,3.5);
  AVAR mean2 = (y2[0] + y2[1] + y2[2]) / 3.0;
  AVAR sum_sq_diff_2 
    = (y2[0] - mean2) * (y2[0] - mean2)
    + (y2[1] - mean2) * (y2[1] - mean2)
    + (y2[2] - mean2) * (y2[2] - mean2); 
  AVAR f2 = sum_sq_diff_2 / (3 - 1);

  EXPECT_EQ(f2.val(), f1_val);

  VEC grad2 = cgrad(f2, y2[0], y2[1], y2[2]);

  EXPECT_EQ(3U, grad1.size());
  EXPECT_EQ(3U, grad2.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(grad2[i], grad1[i]);
}
TEST(AgradRevMatrix, sdStdVector) {
  using stan::math::sd; // should use arg-dep lookup (and for sqrt)

  AVEC y1 = createAVEC(0.5,2.0,3.5);
  AVAR f1 = sd(y1);
  VEC grad1 = cgrad(f1, y1[0], y1[1], y1[2]);
  double f1_val = f1.val(); // save before cleaned out

  AVEC y2 = createAVEC(0.5,2.0,3.5);
  AVAR mean2 = (y2[0] + y2[1] + y2[2]) / 3.0;
  AVAR sum_sq_diff_2 
    = (y2[0] - mean2) * (y2[0] - mean2)
    + (y2[1] - mean2) * (y2[1] - mean2)
    + (y2[2] - mean2) * (y2[2] - mean2); 
  AVAR f2 = sqrt(sum_sq_diff_2 / (3 - 1));

  EXPECT_EQ(f2.val(), f1_val);

  VEC grad2 = cgrad(f2, y2[0], y2[1], y2[2]);

  EXPECT_EQ(3U, grad1.size());
  EXPECT_EQ(3U, grad2.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(grad2[i], grad1[i]);
}


TEST(AgradRevMatrix, initializeVariable) {
  using stan::agrad::initialize_variable;
  using std::vector;

  using Eigen::Matrix;
  using Eigen::Dynamic;

  AVAR a;
  initialize_variable(a, AVAR(1.0));
  EXPECT_FLOAT_EQ(1.0, a.val());

  AVEC b(3);
  initialize_variable(b, AVAR(2.0));
  EXPECT_EQ(3U,b.size());
  EXPECT_FLOAT_EQ(2.0, b[0].val());
  EXPECT_FLOAT_EQ(2.0, b[1].val());
  EXPECT_FLOAT_EQ(2.0, b[2].val());

  vector<AVEC > c(4,AVEC(3));
  initialize_variable(c, AVAR(3.0));
  for (size_t m = 0; m < c.size(); ++m)
    for (size_t n = 0; n < c[0].size(); ++n)
      EXPECT_FLOAT_EQ(3.0,c[m][n].val());

  Matrix<AVAR, Dynamic, Dynamic> aa(5,7);
  initialize_variable(aa, AVAR(4.0));
  for (int m = 0; m < aa.rows(); ++m)
    for (int n = 0; n < aa.cols(); ++n)
      EXPECT_FLOAT_EQ(4.0, aa(m,n).val());

  Matrix<AVAR, Dynamic, 1> bb(5);
  initialize_variable(bb, AVAR(5.0));
  for (int m = 0; m < bb.size(); ++m) 
    EXPECT_FLOAT_EQ(5.0, bb(m).val());

  Matrix<AVAR,1,Dynamic> cc(12);
  initialize_variable(cc, AVAR(7.0));
  for (int m = 0; m < cc.size(); ++m) 
    EXPECT_FLOAT_EQ(7.0, cc(m).val());
  
  Matrix<AVAR,Dynamic,Dynamic> init_val(3,4);
  vector<Matrix<AVAR,Dynamic,Dynamic> > dd(5, init_val);
  initialize_variable(dd, AVAR(11.0));
  for (size_t i = 0; i < dd.size(); ++i)
    for (int m = 0; m < dd[0].rows(); ++m)
      for (int n = 0; n < dd[0].cols(); ++n)
        EXPECT_FLOAT_EQ(11.0, dd[i](m,n).val());
}


TEST(AgradRevMatrix, UserCase1) {
  using std::vector;
  using stan::math::multiply;
  using stan::math::transpose;
  using stan::math::subtract;
  using stan::math::get_base1;
  using stan::math::assign;
  using stan::math::dot_product;
  using stan::agrad::matrix_v;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  // also tried DpKm1 > H
  size_t H = 3;
  size_t DpKm1 = 3;

  vector_v vk(DpKm1);
  for (size_t k = 0; k < DpKm1; ++k)
    vk[k] = (k + 1) * (k + 2);
  
  matrix_v L_etaprec(DpKm1,DpKm1);
  for (size_t m = 0; m < DpKm1; ++m)
    for (size_t n = 0; n < DpKm1; ++n)
      L_etaprec(m,n) = (m + 1) * (n + 1);

  vector_d etamu(DpKm1);
  for (size_t k = 0; k < DpKm1; ++k)
    etamu[k] = 10 + (k * k);
  
  vector<vector_d> eta(H,vector_d(DpKm1));
  for (size_t h = 0; h < H; ++h)
    for (size_t k = 0; k < DpKm1; ++k)
      eta[h][k] = (h + 1) * (k + 10);

  AVAR lp__ = 0.0;

  for (size_t h = 1; h <= H; ++h) {
    assign(vk, multiply(transpose(L_etaprec),
                        subtract(get_base1(eta,h,"eta",1),
                                 etamu)));
    assign(lp__, (lp__ - (0.5 * dot_product(vk,vk))));
  }

  EXPECT_TRUE(lp__.val() != 0);
}

TEST(AgradRevMatrix,prod) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d vd;
  vector_v vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val());

  vd = vector_d(1);
  vv = vector_v(1);
  vd << 2.0;
  vv << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val());

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_v(2);
  vv << 2.0, 3.0;
  AVEC x(2);
  x[0] = vv[0];
  x[1] = vv[1];
  AVAR f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.0,g[0]);
  EXPECT_FLOAT_EQ(2.0,g[1]);
  
}
TEST(AgradRevMatrix,diagMatrix) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_v;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  EXPECT_EQ(0,diag_matrix(vector_v()).size());
  EXPECT_EQ(4,diag_matrix(vector_v(2)).size());
  EXPECT_EQ(0,diag_matrix(vector_d()).size());
  EXPECT_EQ(4,diag_matrix(vector_d(2)).size());

  vector_v v(3);
  v << 1, 4, 9;
  matrix_v m = diag_matrix(v);
  EXPECT_EQ(1,m(0,0).val());
  EXPECT_EQ(4,m(1,1).val());
  EXPECT_EQ(9,m(2,2).val());
}



void test_mult_LLT(const stan::agrad::matrix_v& L) {
  using stan::agrad::matrix_v;
  matrix_v Lp = L; 
  for (int m = 0; m < L.rows(); ++m)
    for (int n = (m+1); n < L.cols(); ++n)
      Lp(m,n) = 0;
  matrix_v LLT_eigen = Lp * Lp.transpose();
  matrix_v LLT_stan = multiply_lower_tri_self_transpose(L);
  EXPECT_EQ(L.rows(),LLT_stan.rows());
  EXPECT_EQ(L.rows(),LLT_stan.cols());
  for (int m = 0; m < L.rows(); ++m)
    for (int n = 0; n < L.rows(); ++n)
      EXPECT_FLOAT_EQ(LLT_eigen(m,n).val(), LLT_stan(m,n).val());  
}

TEST(AgradRevMatrix, multiplyLowerTriSelfTransposeGrad1) {
  using stan::agrad::multiply_lower_tri_self_transpose;
  using stan::agrad::matrix_v;

  matrix_v L(1,1);
  L << 3.0;
  AVEC x(1);
  x[0] = L(0,0);

  matrix_v LLt = multiply_lower_tri_self_transpose(L);
  AVEC y(1);
  y[0] = LLt(0,0);
  
  EXPECT_FLOAT_EQ(9.0, LLt(0,0).val());

  std::vector<VEC > J;
  stan::agrad::jacobian(y,x,J);
  
  EXPECT_FLOAT_EQ(6.0, J[0][0]);
}

TEST(AgradRevMatrix, multiplyLowerTriSelfTransposeGrad2) {
  using stan::agrad::multiply_lower_tri_self_transpose;
  using stan::agrad::matrix_v;

  matrix_v L(2,2);
  L << 
    1, 0,
    2, 3;
  AVEC x(3);
  x[0] = L(0,0);
  x[1] = L(1,0);
  x[2] = L(1,1);

  matrix_v LLt = multiply_lower_tri_self_transpose(L);
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
  stan::agrad::jacobian(y,x,J);

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

TEST(AgradRevMatrix, multiplyLowerTriSelfTransposeGrad3) {
  using stan::agrad::multiply_lower_tri_self_transpose;
  using stan::agrad::matrix_v;
  
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

  matrix_v LLt = multiply_lower_tri_self_transpose(L);
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
  stan::agrad::jacobian(y,x,J);

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

TEST(AgradRevMatrix, multiplyLowerTriSelfTranspose) {
  using stan::agrad::multiply_lower_tri_self_transpose;
  using stan::agrad::matrix_v;
  
  matrix_v L;

  L = matrix_v(3,3);
  L << 1, 0, 0,   
    2, 3, 0,   
    4, 5, 6;
  test_mult_LLT(L);

  L = matrix_v(3,3);
  L << 1, 0, 100000,   
    2, 3, 0,   
    4, 5, 6;
  test_mult_LLT(L);

  L = matrix_v(2,3);
  L << 1, 0, 0,   
    2, 3, 0;   
  test_mult_LLT(L);

  L = matrix_v(3,2);
  L << 1, 0,   
    2, 3,   
    4, 5;
  test_mult_LLT(L);

  matrix_v I(2,2);
  I << 3, 0, 
    4, -3;
  test_mult_LLT(I);

  // matrix_v J(1,1);
  // J << 3.0;
  // test_mult_LLT(J);

  // matrix_v K(0,0);
  // test_mult_LLT(K);
}

void test_tcrossprod(const stan::agrad::matrix_v& L) {
  using stan::agrad::matrix_v;
  using stan::agrad::tcrossprod;
  matrix_v LLT_eigen = L * L.transpose();
  matrix_v LLT_stan = tcrossprod(L);
  EXPECT_EQ(LLT_eigen.rows(),LLT_stan.rows());
  EXPECT_EQ(LLT_eigen.rows(),LLT_stan.cols());
  for (int m = 0; m < LLT_eigen.rows(); ++m)
    for (int n = 0; n < LLT_eigen.cols(); ++n)
      EXPECT_FLOAT_EQ(LLT_eigen(m,n).val(), LLT_stan(m,n).val());
}
TEST(AgradRevMatrix, tcrossprod) {
  using stan::agrad::matrix_v;

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
  using stan::agrad::tcrossprod;
  using stan::agrad::matrix_v;

  matrix_v L(1,1);
  L << 3.0;
  AVEC x(1);
  x[0] = L(0,0);

  matrix_v LLt = tcrossprod(L);
  AVEC y(1);
  y[0] = LLt(0,0);

  EXPECT_FLOAT_EQ(9.0, LLt(0,0).val());

  std::vector<VEC > J;
  stan::agrad::jacobian(y,x,J);

  EXPECT_FLOAT_EQ(6.0, J[0][0]);
}

TEST(AgradRevMatrix, tcrossprodGrad2) {
  using stan::agrad::tcrossprod;
  using stan::agrad::matrix_v;

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
  stan::agrad::jacobian(y,x,J);

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
  using stan::agrad::tcrossprod;
  using stan::agrad::matrix_v;

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
  stan::agrad::jacobian(y,x,J);

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

template <typename T>
void test_cumulative_sum() {
  using stan::math::cumulative_sum;

  T c(1);
  c[0] = 1.7;
  T d = cumulative_sum(c);
  EXPECT_EQ(c.size(), d.size());
  EXPECT_FLOAT_EQ(c[0].val(),d[0].val());
  VEC grad = cgrad(d[0], c[0]);
  EXPECT_EQ(1U,grad.size());
  EXPECT_FLOAT_EQ(1.0,grad[0]);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  T f = cumulative_sum(e);
  EXPECT_EQ(e.size(), f.size());
  EXPECT_FLOAT_EQ(e[0].val(),f[0].val());
  EXPECT_FLOAT_EQ((e[0] + e[1]).val(), f[1].val());
  grad = cgrad(f[0],e[0],e[1]);
  EXPECT_EQ(2U,grad.size());
  EXPECT_FLOAT_EQ(1.0,grad[0]);
  EXPECT_FLOAT_EQ(0.0,grad[1]);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  T h = cumulative_sum(g);
  EXPECT_EQ(g.size(), h.size());
  EXPECT_FLOAT_EQ(g[0].val(),h[0].val());
  EXPECT_FLOAT_EQ((g[0] + g[1]).val(), h[1].val());
  EXPECT_FLOAT_EQ((g[0] + g[1] + g[2]).val(), h[2].val());

  grad = cgrad(h[2],g[0],g[1],g[2]);
  EXPECT_EQ(3U,grad.size());
  EXPECT_FLOAT_EQ(1.0,grad[0]);
  EXPECT_FLOAT_EQ(1.0,grad[1]);
  EXPECT_FLOAT_EQ(1.0,grad[2]);
}
TEST(AgradRevMatrix, cumulative_sum) {
  using stan::agrad::var;
  using stan::math::cumulative_sum;

  EXPECT_FLOAT_EQ(0, cumulative_sum(std::vector<var>(0)).size());

  Eigen::Matrix<var,Eigen::Dynamic,1> a;
  EXPECT_FLOAT_EQ(0,cumulative_sum(a).size());

  Eigen::Matrix<var,1,Eigen::Dynamic> b;
  EXPECT_FLOAT_EQ(0,cumulative_sum(b).size());

  test_cumulative_sum<std::vector<var> >();
  test_cumulative_sum<Eigen::Matrix<var,Eigen::Dynamic,1> >();
  test_cumulative_sum<Eigen::Matrix<var,1,Eigen::Dynamic> >();
}

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



