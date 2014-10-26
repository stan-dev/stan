#include <stan/agrad/rev/matrix/mdivide_left_tri.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/mdivide_left_tri.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradRevMatrix,mdivide_left_tri_val) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::mdivide_left_tri;

  matrix_v Av(2,2);
  matrix_v Av_inv(2,2);
  matrix_d Ad(2,2);
  matrix_v I;
  
  Av << 2.0, 0.0,
    5.0, 7.0;
  Ad << 2.0, 0.0, 
    5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left_tri<Eigen::Lower>(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left_tri<Eigen::Lower>(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  Av_inv = mdivide_left_tri<Eigen::Lower>(Av);
  I = Av * Av_inv;
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  Av << 2.0, 3.0, 
    0.0, 7.0;
  Ad << 2.0, 3.0, 
    0.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left_tri<Eigen::Upper>(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left_tri<Eigen::Upper>(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);
}
TEST(AgradRevMatrix,mdivide_left_tri_lower_grad_vv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::mdivide_left_tri;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 0.0, 
  5.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k,l;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v A(2,2);
      matrix_v B(2,2);
      matrix_v C;
      VEC g;

      for (k = 0; k < 4; k++) {
        A(k) = Ad(k);
        B(k) = Bd(k);
      }
      
      C = mdivide_left_tri<Eigen::Lower>(A,B);
      AVEC x = createAVEC(A(0,0),A(1,0),A(0,1),A(1,1),
                          B(0,0),B(1,0),B(0,1),B(1,1));
      C(i,j).grad(x,g);
      
      for (l = 0; l < Ad_tmp.cols(); l++) {
        for (k = 0; k < Ad_tmp.rows(); k++) {
          if (k >= l) {
            Ad_tmp.setZero();
            Ad_tmp(k,l) = 1.0;
            Cd = -mdivide_left_tri<Eigen::Lower>(Ad,multiply(Ad_tmp,mdivide_left_tri<Eigen::Lower>(Ad,Bd)));
            EXPECT_NEAR(Cd(i,j),g[k + l*Ad_tmp.rows()],1.0E-12);
          }
          else {
            EXPECT_NEAR(0.0,g[k + l*Ad_tmp.rows()],1.0E-12);
          }
        }
      }
      for (k = 0; k < 4; k++) {
        Bd_tmp.setZero();
        Bd_tmp(k) = 1.0;
        Cd = mdivide_left_tri<Eigen::Lower>(Ad,Bd_tmp);
        EXPECT_NEAR(Cd(i,j),g[4+k],1.0E-12);
      }
    }
  }
}

TEST(AgradRevMatrix,mdivide_left_tri_lower_grad_dv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::mdivide_left_tri;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 0.0, 
  5.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v B(2,2);
      matrix_v C;
      VEC g;
      
      for (k = 0; k < 4; k++) {
        B(k) = Bd(k);
      }
      
      C = mdivide_left_tri<Eigen::Lower>(Ad,B);
      AVEC x = createAVEC(B(0,0),B(1,0),B(0,1),B(1,1));
      C(i,j).grad(x,g);

      for (k = 0; k < 4; k++) {
        Bd_tmp.setZero();
        Bd_tmp(k) = 1.0;
        Cd = mdivide_left_tri<Eigen::Lower>(Ad,Bd_tmp);
        EXPECT_NEAR(Cd(i,j),g[k],1.0E-12);
      }
    }
  }
}

TEST(AgradRevMatrix,mdivide_left_tri_lower_grad_vd) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::mdivide_left_tri;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 0.0, 
  5.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k,l;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v A(2,2);
      matrix_v C;
      VEC g;
      
      for (k = 0; k < 4; k++) {
        A(k) = Ad(k);
      }
      
      C = mdivide_left_tri<Eigen::Lower>(A,Bd);
      AVEC x = createAVEC(A(0,0),A(1,0),A(0,1),A(1,1));
      C(i,j).grad(x,g);
      
      for (l = 0; l < Ad_tmp.cols(); l++) {
        for (k = 0; k < Ad_tmp.rows(); k++) {
          if (k >= l) {
            Ad_tmp.setZero();
            Ad_tmp(k,l) = 1.0;
            Cd = -mdivide_left_tri<Eigen::Lower>(Ad,multiply(Ad_tmp,mdivide_left_tri<Eigen::Lower>(Ad,Bd)));
            EXPECT_NEAR(Cd(i,j),g[k + l*Ad_tmp.rows()],1.0E-12);
          }
          else {
            EXPECT_NEAR(0.0,g[k + l*Ad_tmp.rows()],1.0E-12);
          }
        }
      }
    }
  }
}
TEST(AgradRevMatrix,mdivide_left_tri_upper_grad_vv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::mdivide_left_tri;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  0.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k,l;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v A(2,2);
      matrix_v B(2,2);
      matrix_v C;
      VEC g;
      
      for (k = 0; k < 4; k++) {
        A(k) = Ad(k);
        B(k) = Bd(k);
      }
      
      C = mdivide_left_tri<Eigen::Upper>(A,B);
      AVEC x = createAVEC(A(0,0),A(1,0),A(0,1),A(1,1),
                          B(0,0),B(1,0),B(0,1),B(1,1));
      C(i,j).grad(x,g);
      
      for (l = 0; l < Ad_tmp.cols(); l++) {
        for (k = 0; k < Ad_tmp.rows(); k++) {
          if (k <= l) {
            Ad_tmp.setZero();
            Ad_tmp(k,l) = 1.0;
            Cd = -mdivide_left_tri<Eigen::Upper>(Ad,multiply(Ad_tmp,mdivide_left_tri<Eigen::Upper>(Ad,Bd)));
            EXPECT_NEAR(Cd(i,j),g[k + l*Ad_tmp.rows()],1.0E-12);
          }
          else {
            EXPECT_NEAR(0.0,g[k + l*Ad_tmp.rows()],1.0E-12);
          }
        }
      }
      for (k = 0; k < 4; k++) {
        Bd_tmp.setZero();
        Bd_tmp(k) = 1.0;
        Cd = mdivide_left_tri<Eigen::Upper>(Ad,Bd_tmp);
        EXPECT_NEAR(Cd(i,j),g[4+k],1.0E-12);
      }
    }
  }
}

TEST(AgradRevMatrix,mdivide_left_tri_upper_grad_dv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::mdivide_left_tri;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  0.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v B(2,2);
      matrix_v C;
      VEC g;
      
      for (k = 0; k < 4; k++) {
        B(k) = Bd(k);
      }
      
      C = mdivide_left_tri<Eigen::Upper>(Ad,B);
      AVEC x = createAVEC(B(0,0),B(1,0),B(0,1),B(1,1));
      C(i,j).grad(x,g);
      
      for (k = 0; k < 4; k++) {
        Bd_tmp.setZero();
        Bd_tmp(k) = 1.0;
        Cd = mdivide_left_tri<Eigen::Upper>(Ad,Bd_tmp);
        EXPECT_NEAR(Cd(i,j),g[k],1.0E-12);
      }
    }
  }
}

TEST(AgradRevMatrix,mdivide_left_tri_upper_grad_vd) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::mdivide_left_tri;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  0.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k,l;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v A(2,2);
      matrix_v C;
      VEC g;
      
      for (k = 0; k < 4; k++) {
        A(k) = Ad(k);
      }
      
      C = mdivide_left_tri<Eigen::Upper>(A,Bd);
      AVEC x = createAVEC(A(0,0),A(1,0),A(0,1),A(1,1));
      C(i,j).grad(x,g);
      
      for (l = 0; l < Ad_tmp.cols(); l++) {
        for (k = 0; k < Ad_tmp.rows(); k++) {
          if (k <= l) {
            Ad_tmp.setZero();
            Ad_tmp(k,l) = 1.0;
            Cd = -mdivide_left_tri<Eigen::Upper>(Ad,multiply(Ad_tmp,mdivide_left_tri<Eigen::Upper>(Ad,Bd)));
            EXPECT_NEAR(Cd(i,j),g[k + l*Ad_tmp.rows()],1.0E-12);
          }
          else {
            EXPECT_NEAR(0.0,g[k + l*Ad_tmp.rows()],1.0E-12);
          }
        }
      }
    }
  }
}
// // FIXME:  Fails in g++ 4.2 -- can't find agrad version of mdivide_left_tri
// //         Works in clang++ and later g++
// // TEST(AgradRevMatrix,mdivide_left_tri2) {
// //   using stan::math::mdivide_left_tri;
// //   using stan::agrad::mdivide_left_tri;
// //   int k = 3;
// //   Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic> L(k,k);
// //   L << 1, 2, 3, 4, 5, 6, 7, 8, 9;
// //   Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic> I(k,k);
// //   I.setIdentity();
// //   L = mdivide_left_tri<Eigen::Lower>(L, I);
// // }
