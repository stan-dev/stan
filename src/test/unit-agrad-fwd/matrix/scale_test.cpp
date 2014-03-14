#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/scale.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

using stan::math::matrix_d;
using stan::math::vector_d;
using stan::math::row_vector_d;
using stan::math::scale;

using stan::agrad::matrix_fd;
using stan::agrad::fvar;
using stan::agrad::vector_fd;
using stan::agrad::row_vector_fd;

void test_fwd_scale_vv(matrix_d md, vector_d vd) {
  int M = md.rows();
  int N = md.cols();
  int K = vd.size();
  matrix_d md_vd = scale(md,vd);

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
      matrix_fd mf_vf = scale(mf, vf);
      EXPECT_EQ(M, mf_vf.rows());
      EXPECT_EQ(N, mf_vf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_vf(m,n).val_);

      // left tangent
      EXPECT_FLOAT_EQ(2.5 * vd(n1) * vd(m1), mf_vf(m1,n1).d_);

      // SAME FOR ROW VECTORS
      row_vector_fd rvf(K);
      for (int k = 0; k < K; ++k)
        rvf(k).val_ = vd(k);

      // value
      matrix_fd mf_rvf = scale(mf, rvf);
      EXPECT_EQ(M, mf_rvf.rows());
      EXPECT_EQ(N, mf_rvf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_rvf(m,n).val_);

      // left tangent
      EXPECT_FLOAT_EQ(2.5 * vd(n1) * vd(m1), mf_rvf(m1,n1).d_);

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

      matrix_fd mf_vf = scale(mf, vf);

      // right tangent
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          if (n == k1 && m == k1) 
            EXPECT_FLOAT_EQ(3.7 * 2 * vd(m) * md(m,n), mf_vf(m,n).d_);
          else if (n == k1) 
            EXPECT_FLOAT_EQ(3.7 * vd(m) * md(m,n), mf_vf(m,n).d_);
          else if (m == k1) 
            EXPECT_FLOAT_EQ(3.7 * vd(n) * md(m,n), mf_vf(m,n).d_);
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

      matrix_fd mf_rvf = scale(mf, rvf);

      // right tangent
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          if (n == k1 && m == k1) 
            EXPECT_FLOAT_EQ(3.7 * 2 * vd(m) * md(m,n), mf_rvf(m,n).d_);
          else if (n == k1) 
            EXPECT_FLOAT_EQ(3.7 * vd(m) * md(m,n), mf_rvf(m,n).d_);
          else if (m == k1) 
            EXPECT_FLOAT_EQ(3.7 * vd(n) * md(m,n), mf_rvf(m,n).d_);
          else
            EXPECT_FLOAT_EQ(0.0, mf_rvf(m,n).d_);
       }
      }
  } 
}
void test_fwd_scale_vd(matrix_d md, vector_d vd) {
  int M = md.rows();
  int N = md.cols();
  int K = vd.size();
  matrix_d md_vd = scale(md,vd);

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
      matrix_fd mf_vf = scale(mf, vd);
      EXPECT_EQ(M, mf_vf.rows());
      EXPECT_EQ(N, mf_vf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_vf(m,n).val_);

      // left tangent
      EXPECT_FLOAT_EQ(2.5 * vd(n1) * vd(m1), mf_vf(m1,n1).d_);
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
      matrix_fd mf_rvf = scale(mf, rvf);
      EXPECT_EQ(M, mf_rvf.rows());
      EXPECT_EQ(N, mf_rvf.cols());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
          EXPECT_FLOAT_EQ(md_vd(m,n), mf_rvf(m,n).val_);

      // left tangent
      EXPECT_FLOAT_EQ(2.5 * vd(n1) * vd(m1), mf_rvf(m1,n1).d_);
    }
  }

}
void test_fwd_scale_dv(matrix_d md, vector_d vd) {
  int M = md.rows();
  int N = md.cols();
  int K = vd.size();
  matrix_d md_vd = scale(md,vd);

  // value
  for (int m1 = 0; m1 < M; ++m1) {
    for (int n1 = 0; n1 < N; ++n1) { 

      vector_fd vf(K);
      for (int k = 0; k < K; ++k)
        vf(k).val_ = vd(k);

      // value
      matrix_fd mf_vf = scale(md, vf);
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
      matrix_fd mf_rvf = scale(md, rvf);
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

      matrix_fd mf_vf = scale(md, vf);

      // right tangent
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          if (n == k1 && m == k1) 
            EXPECT_FLOAT_EQ(3.7 * 2 * vd(m) * md(m,n), mf_vf(m,n).d_);
          else if (n == k1) 
            EXPECT_FLOAT_EQ(3.7 * vd(m) * md(m,n), mf_vf(m,n).d_);
          else if (m == k1) 
            EXPECT_FLOAT_EQ(3.7 * vd(n) * md(m,n), mf_vf(m,n).d_);
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

      matrix_fd mf_rvf = scale(md, rvf);

      // right tangent
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          if (n == k1 && m == k1) 
            EXPECT_FLOAT_EQ(3.7 * 2 * vd(m) * md(m,n), mf_rvf(m,n).d_);
          else if (n == k1) 
            EXPECT_FLOAT_EQ(3.7 * vd(m) * md(m,n), mf_rvf(m,n).d_);
          else if (m == k1) 
            EXPECT_FLOAT_EQ(3.7 * vd(n) * md(m,n), mf_rvf(m,n).d_);
          else
            EXPECT_FLOAT_EQ(0.0, mf_rvf(m,n).d_);
        }
      }
  }
}

void test_fwd_scale(matrix_d m, vector_d v) {
  test_fwd_scale_vv(m,v);
  test_fwd_scale_vd(m,v);
  test_fwd_scale_dv(m,v);
}

TEST(AgradFwdMatrixScale, ff1) {
  matrix_d m(1,1);
  m << 10;
  vector_d v(1);
  v << 3;
  test_fwd_scale(m,v);
}
TEST(AgradFwdMatrixScale, ff2) {
  matrix_d m(2,2);
  m << 1, 10, 100, 1000;
  vector_d v(2);
  v << 2, 3;
  test_fwd_scale(m,v);
}

TEST(AgradFwdMatrixScale, ff3) {
  matrix_d m(3,3);
  m << 1, 10, 100, 1000, 2, -4, 8, -16, 32;
  vector_d v(3);
  v << -1.7, 111.2, -29.3;
  test_fwd_scale(m,v);
}
TEST(AgradFwdMatrixScale, exceptions) {
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

  EXPECT_THROW(scale(mf,mf), std::domain_error);

  EXPECT_THROW(scale(mf,v), std::domain_error);
  EXPECT_THROW(scale(m,vf), std::domain_error);
  EXPECT_THROW(scale(mf,vf), std::domain_error);

  EXPECT_THROW(scale(mf,rv), std::domain_error);
  EXPECT_THROW(scale(m,rvf), std::domain_error);
  EXPECT_THROW(scale(mf,rvf), std::domain_error);
}
