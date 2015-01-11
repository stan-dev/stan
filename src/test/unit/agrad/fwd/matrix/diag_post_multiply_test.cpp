#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/diag_post_multiply.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

using stan::math::matrix_d;
using stan::math::vector_d;
using stan::math::row_vector_d;
using stan::math::diag_post_multiply;

using stan::agrad::matrix_fd;
using stan::agrad::fvar;
using stan::agrad::vector_fd;
using stan::agrad::row_vector_fd;

void test_fwd_diag_post_multiply_vv(matrix_d md, vector_d vd) {
  int M = md.rows();
  int N = md.cols();
  int K = vd.size();
  matrix_d md_vd = diag_post_multiply(md,vd);

  // left tangent & value
  for (int m1 = 0; m1 < M; ++m1) {
    for (int n1 = 0; n1 < N; ++n1) { 

      matrix_fd mf(M,N);
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          mf(m,n).val_ = md(m,n);
          if (m == m1 && n == n1)
            mf(m,n).d_ = 2.5;
        }
      }

      vector_fd vf(K);
      for (int k = 0; k < K; ++k)
        vf(k).val_ = vd(k);

      // value
      matrix_fd mf_vf = diag_post_multiply(mf, vf);
      EXPECT_EQ(M, mf_vf.rows());
      EXPECT_EQ(N, mf_vf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_vf(m,n).val_);

      // left tangent
      EXPECT_FLOAT_EQ(2.5 * vd(n1), mf_vf(m1,n1).d_);

      // SAME FOR ROW VECTORS
      row_vector_fd rvf(K);
      for (int k = 0; k < K; ++k)
        rvf(k).val_ = vd(k);

      // value
      matrix_fd mf_rvf = diag_post_multiply(mf, rvf);
      EXPECT_EQ(M, mf_rvf.rows());
      EXPECT_EQ(N, mf_rvf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_rvf(m,n).val_);

      // left tangent
      EXPECT_FLOAT_EQ(2.5 * vd(n1), mf_rvf(m1,n1).d_);

    }
  }
  
  // right tangent
  for (int k1 = 0; k1 < K; ++k1) {

      matrix_fd mf(M,N);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          mf(m,n).val_ = md(m,n);

      vector_fd vf(K);
      for (int k = 0; k < K; ++k) {
        vf(k).val_ = vd(k);
        if (k == k1) vf(k).d_ = 3.7;
      }

      matrix_fd mf_vf = diag_post_multiply(mf, vf);

      // right tangent
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          if (n == k1) 
            EXPECT_FLOAT_EQ(3.7 * md(m,n), mf_vf(m,n).d_);
          else
            EXPECT_FLOAT_EQ(0.0, mf_vf(m,n).d_);
        }
      }
      
      // SAME FOR ROW VECTORS
      row_vector_fd rvf(K);
      for (int k = 0; k < K; ++k) {
        rvf(k).val_ = vd(k);
        if (k == k1) rvf(k).d_ = 3.7;
      }

      matrix_fd mf_rvf = diag_post_multiply(mf, rvf);

      // right tangent
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          if (n == k1) 
            EXPECT_FLOAT_EQ(3.7 * md(m,n), mf_rvf(m,n).d_);
          else
            EXPECT_FLOAT_EQ(0.0, mf_rvf(m,n).d_);
        }
      }
  }
}
void test_fwd_diag_post_multiply_vd(matrix_d md, vector_d vd) {
  int M = md.rows();
  int N = md.cols();
  int K = vd.size();
  matrix_d md_vd = diag_post_multiply(md,vd);

  // left tangent & value
  for (int m1 = 0; m1 < M; ++m1) {
    for (int n1 = 0; n1 < N; ++n1) { 

      matrix_fd mf(M,N);
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          mf(m,n).val_ = md(m,n);
          if (m == m1 && n == n1)
            mf(m,n).d_ = 2.5;
        }
      }

      // value
      matrix_fd mf_vf = diag_post_multiply(mf, vd);
      EXPECT_EQ(M, mf_vf.rows());
      EXPECT_EQ(N, mf_vf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_vf(m,n).val_);

      // left tangent
      EXPECT_FLOAT_EQ(2.5 * vd(n1), mf_vf(m1,n1).d_);
    }
  }

  // SAME FOR ROW VECTORS
  row_vector_d rvf(K);
  for (int k = 0; k < K; ++k)
    rvf(k) = vd(k);
  for (int m1 = 0; m1 < M; ++m1) {
    for (int n1 = 0; n1 < N; ++n1) { 

      matrix_fd mf(M,N);
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          mf(m,n).val_ = md(m,n);
          if (m == m1 && n == n1)
            mf(m,n).d_ = 2.5;
        }
      }

      // value
      matrix_fd mf_rvf = diag_post_multiply(mf, rvf);
      EXPECT_EQ(M, mf_rvf.rows());
      EXPECT_EQ(N, mf_rvf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_rvf(m,n).val_);

      // left tangent
      EXPECT_FLOAT_EQ(2.5 * vd(n1), mf_rvf(m1,n1).d_);
    }
  }

}
void test_fwd_diag_post_multiply_dv(matrix_d md, vector_d vd) {
  int M = md.rows();
  int N = md.cols();
  int K = vd.size();
  matrix_d md_vd = diag_post_multiply(md,vd);

  // value
  for (int m1 = 0; m1 < M; ++m1) {
    for (int n1 = 0; n1 < N; ++n1) { 

      vector_fd vf(K);
      for (int k = 0; k < K; ++k)
        vf(k).val_ = vd(k);

      // value
      matrix_fd mf_vf = diag_post_multiply(md, vf);
      EXPECT_EQ(M, mf_vf.rows());
      EXPECT_EQ(N, mf_vf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_vf(m,n).val_);

      // ROW VECTOR VALUE
      row_vector_fd rvf(K);
      for (int k = 0; k < K; ++k)
        rvf(k).val_ = vd(k);

      // value
      matrix_fd mf_rvf = diag_post_multiply(md, rvf);
      EXPECT_EQ(M, mf_rvf.rows());
      EXPECT_EQ(N, mf_rvf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_rvf(m,n).val_);
    }
  }
  
  // right tangent
  for (int k1 = 0; k1 < K; ++k1) {

      vector_fd vf(K);
      for (int k = 0; k < K; ++k) {
        vf(k).val_ = vd(k);
        if (k == k1) vf(k).d_ = 3.7;
      }

      matrix_fd mf_vf = diag_post_multiply(md, vf);

      // right tangent
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          if (n == k1) 
            EXPECT_FLOAT_EQ(3.7 * md(m,n), mf_vf(m,n).d_);
          else
            EXPECT_FLOAT_EQ(0.0, mf_vf(m,n).d_);
        }
      }

      // ROW VECTORS
      row_vector_fd rvf(K);
      for (int k = 0; k < K; ++k) {
        rvf(k).val_ = vd(k);
        if (k == k1) rvf(k).d_ = 3.7;
      }

      matrix_fd mf_rvf = diag_post_multiply(md, rvf);

      // right tangent
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          if (n == k1) 
            EXPECT_FLOAT_EQ(3.7 * md(m,n), mf_rvf(m,n).d_);
          else
            EXPECT_FLOAT_EQ(0.0, mf_rvf(m,n).d_);
        }
      }
  }
}

void test_fwd_diag_post_multiply(matrix_d m, vector_d v) {
  test_fwd_diag_post_multiply_vv(m,v);
  test_fwd_diag_post_multiply_vd(m,v);
  test_fwd_diag_post_multiply_dv(m,v);
}

TEST(AgradFwdMatrixDiagPostMultiply, ff1) {
  matrix_d m(1,1);
  m << 10;
  vector_d v(1);
  v << 3;
  test_fwd_diag_post_multiply(m,v);
}

TEST(AgradFwdMatrixDiagPostMultiply, ff2) {
  matrix_d m(2,2);
  m << 1, 10, 100, 1000;
  vector_d v(2);
  v << 2, 3;
  test_fwd_diag_post_multiply(m,v);
}

TEST(AgradFwdMatrixDiagPostMultiply, ff3) {
  matrix_d m(3,3);
  m << 1, 10, 100, 1000, 2, -4, 8, -16, 32;
  vector_d v(3);
  v << -1.7, 111.2, -29.3;
  test_fwd_diag_post_multiply(m,v);
}

TEST(AgradFwdMatrixDiagPostMultiply, exceptions) {
  matrix_d m(3,3);
  m << 1, 10, 100, 1000, 2, -4, 8, -16, 32;

  vector_d v(2);
  v << -1.7, 111.2;

  vector_d rv(2);
  v << -1.7, 111.2;

  matrix_fd mf(3,3);
  mf << 1, 10, 100, 1000, 2, -4, 8, -16, 32;

  vector_fd vf(2);
  vf << -1.7, 111.2;

  vector_fd rvf(2);
  rvf << -1.7, 111.2;

  EXPECT_THROW(diag_post_multiply(mf,mf), std::domain_error);

  EXPECT_THROW(diag_post_multiply(mf,v), std::domain_error);
  EXPECT_THROW(diag_post_multiply(m,vf), std::domain_error);
  EXPECT_THROW(diag_post_multiply(mf,vf), std::domain_error);

  EXPECT_THROW(diag_post_multiply(mf,rv), std::domain_error);
  EXPECT_THROW(diag_post_multiply(m,rvf), std::domain_error);
  EXPECT_THROW(diag_post_multiply(mf,rvf), std::domain_error);
}

TEST(AgradFwdMatrixDiagPostMultiply, vector_fv_1stDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);

  matrix_fv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  vector_d A(3);
  A << 1, 2, 3;
  vector_fv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_fv output = stan::math::diag_post_multiply(Y,B);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}

TEST(AgradFwdMatrixDiagPostMultiply, vector_fv_2ndDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);

  matrix_fv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  vector_fv B(3);
  B << a,b,c;

  matrix_fv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, vector_fv_exception) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  vector_fv B(3);
  vector_fv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}

TEST(AgradFwdMatrixDiagPostMultiply, rowvector_fv_1stDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);

  matrix_fv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_fv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_fv output = stan::math::diag_post_multiply(Y,B);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_fv_2ndDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);

  matrix_fv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  row_vector_fv B(3);
  B << a,b,c;

  matrix_fv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_fv_exception) {
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;

  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  row_vector_fv B(3);
  row_vector_fv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPostMultiply, vector_ffd) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffd;
  using stan::math::vector_d;
  using stan::agrad::fvar;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 2.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 2.0;
  f.val_.val_ = 6.0;
  f.d_.val_ = 2.0;

  matrix_ffd Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  vector_d A(3);
  A << 1, 2, 3;
  vector_ffd B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_ffd output = stan::math::diag_post_multiply(Y,B);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val());
}
TEST(AgradFwdMatrixDiagPostMultiply, vector_ffd_exception) {
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  matrix_ffd Y(3,3);
  matrix_ffd Z(2,3);
  vector_ffd B(3);
  vector_ffd C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}

TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffd) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_ffd;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 2.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 2.0;
  f.val_.val_ = 6.0;
  f.d_.val_ = 2.0;

  matrix_ffd Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_ffd B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_ffd output = stan::math::diag_post_multiply(Y,B);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val());
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffd_exception) {
  using stan::agrad::matrix_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  matrix_ffd Y(3,3);
  matrix_ffd Z(2,3);
  row_vector_ffd B(3);
  row_vector_ffd C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}

TEST(AgradFwdMatrixDiagPostMultiply, vector_ffv_1stDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  vector_d A(3);
  A << 1, 2, 3;
  vector_ffv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, vector_ffv_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}

TEST(AgradFwdMatrixDiagPostMultiply, vector_ffv_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, vector_ffv_3rdDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::math::vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(5.0,1.0);
  fvar<fvar<var> > f(6.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, vector_ffv_exception) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;

  matrix_ffv Y(3,3);
  matrix_ffv Z(2,3);
  vector_ffv B(3);
  vector_ffv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}

TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffv_1stDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffv_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffv_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffv_3rdDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(5.0,1.0);
  fvar<fvar<var> > f(6.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffv_exception) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;

  matrix_ffv Y(3,3);
  matrix_ffv Z(2,3);
  row_vector_ffv B(3);
  row_vector_ffv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
