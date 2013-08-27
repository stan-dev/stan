#include <stan/diff/rev/matrix/mdivide_left_spd.hpp>
#include <gtest/gtest.h>
#include <test/diff/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/mdivide_left_spd.hpp>
#include <stan/math/matrix/multiply.hpp>

TEST(DiffRevMatrix,mdivide_left_spd_val) {
  using stan::math::matrix_d;
  using stan::diff::matrix_v;
  using stan::math::mdivide_left_spd;

  matrix_v Av(2,2);
  matrix_d Ad(2,2);
  matrix_v I;

  Av << 2.0, 3.0, 
    3.0, 7.0;
  Ad << 2.0, 3.0, 
    3.0, 7.0;

  I = mdivide_left_spd(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left_spd(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left_spd(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);
}

TEST(DiffRevMatrix,mdivide_left_spd_grad_vv) {
  using stan::math::matrix_d;
  using stan::diff::matrix_v;
  using stan::math::mdivide_left_spd;
  using stan::math::multiply;
  

  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  3.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v A(2,2);
      matrix_v B(2,2);
      matrix_v C;

      for (k = 0; k < 4; k++) {
        A(k) = Ad(k);
        B(k) = Bd(k);
      }
      
      C = mdivide_left_spd(A,B);
      AVEC x = createAVEC(A(0,0),A(1,0),A(0,1),A(1,1),
                          B(0,0),B(1,0),B(0,1),B(1,1));

      
      VEC g;
      C(i,j).grad(x,g);
      
      for (k = 0; k < 4; k++) {
        Ad_tmp.setZero();
        Ad_tmp(k) = 1.0;
        Cd = -mdivide_left_spd(Ad,multiply(Ad_tmp,mdivide_left_spd(Ad,Bd)));
        EXPECT_NEAR(Cd(i,j),g[k],1.0E-12);
      }
      for (k = 0; k < 4; k++) {
        Bd_tmp.setZero();
        Bd_tmp(k) = 1.0;
        Cd = mdivide_left_spd(Ad,Bd_tmp);
        EXPECT_NEAR(Cd(i,j),g[4+k],1.0E-12);
      }
    }
  }
}

TEST(DiffRevMatrix,mdivide_left_spd_grad_dv) {
  using stan::math::matrix_d;
  using stan::diff::matrix_v;
  using stan::math::mdivide_left_spd;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  3.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v B(2,2);
      matrix_v C;
      
      for (k = 0; k < 4; k++) {
        B(k) = Bd(k);
      }
      
      C = mdivide_left_spd(Ad,B);
      AVEC x = createAVEC(B(0,0),B(1,0),B(0,1),B(1,1));
      
      
      VEC g;
      C(i,j).grad(x,g);

      for (k = 0; k < 4; k++) {
        Bd_tmp.setZero();
        Bd_tmp(k) = 1.0;
        Cd = mdivide_left_spd(Ad,Bd_tmp);
        EXPECT_NEAR(Cd(i,j),g[k],1.0E-12);
      }
    }
  }
}

TEST(DiffRevMatrix,mdivide_left_spd_grad_vd) {
  using stan::math::matrix_d;
  using stan::diff::matrix_v;
  using stan::math::mdivide_left_spd;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  3.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;
  
  size_type i,j,k;
  for (i = 0; i < Bd.rows(); i++) {
    for (j = 0; j < Bd.cols(); j++) {
      matrix_v A(2,2);
      matrix_v C;
      
      for (k = 0; k < 4; k++) {
        A(k) = Ad(k);
      }
      
      C = mdivide_left_spd(A,Bd);
      AVEC x = createAVEC(A(0,0),A(1,0),A(0,1),A(1,1));
      
      
      VEC g;
      C(i,j).grad(x,g);
      
      for (k = 0; k < 4; k++) {
        Ad_tmp.setZero();
        Ad_tmp(k) = 1.0;
        Cd = -mdivide_left_spd(Ad,multiply(Ad_tmp,mdivide_left_spd(Ad,Bd)));
        EXPECT_NEAR(Cd(i,j),g[k],1.0E-12);
      }
    }
  }
}
