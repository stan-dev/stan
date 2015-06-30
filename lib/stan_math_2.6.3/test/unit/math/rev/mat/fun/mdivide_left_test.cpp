#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/mdivide_left.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/mdivide_left.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>

TEST(AgradRevMatrix,mdivide_left_val) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::mdivide_left;

  matrix_v Av(2,2);
  matrix_d Ad(2,2);
  matrix_v I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_left(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);
}

TEST(AgradRevMatrix,mdivide_left_grad_vv) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::mdivide_left;
  using stan::math::multiply;
  

  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  5.0, 7.0;
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
      
      C = mdivide_left(A,B);
      AVEC x = createAVEC(A(0,0),A(1,0),A(0,1),A(1,1),
                          B(0,0),B(1,0),B(0,1),B(1,1));

      
      VEC g;
      C(i,j).grad(x,g);
      
      for (k = 0; k < 4; k++) {
        Ad_tmp.setZero();
        Ad_tmp(k) = 1.0;
        Cd = -mdivide_left(Ad,multiply(Ad_tmp,mdivide_left(Ad,Bd)));
        EXPECT_NEAR(Cd(i,j),g[k],1.0E-12);
      }
      for (k = 0; k < 4; k++) {
        Bd_tmp.setZero();
        Bd_tmp(k) = 1.0;
        Cd = mdivide_left(Ad,Bd_tmp);
        EXPECT_NEAR(Cd(i,j),g[4+k],1.0E-12);
      }
    }
  }
}

TEST(AgradRevMatrix,mdivide_left_grad_dv) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::mdivide_left;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  5.0, 7.0;
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
      
      C = mdivide_left(Ad,B);
      AVEC x = createAVEC(B(0,0),B(1,0),B(0,1),B(1,1));
      
      
      VEC g;
      C(i,j).grad(x,g);

      for (k = 0; k < 4; k++) {
        Bd_tmp.setZero();
        Bd_tmp(k) = 1.0;
        Cd = mdivide_left(Ad,Bd_tmp);
        EXPECT_NEAR(Cd(i,j),g[k],1.0E-12);
      }
    }
  }
}

TEST(AgradRevMatrix,mdivide_left_grad_vd) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::mdivide_left;
  using stan::math::multiply;
  
  
  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  5.0, 7.0;
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
      
      C = mdivide_left(A,Bd);
      AVEC x = createAVEC(A(0,0),A(1,0),A(0,1),A(1,1));
      
      
      VEC g;
      C(i,j).grad(x,g);
      
      for (k = 0; k < 4; k++) {
        Ad_tmp.setZero();
        Ad_tmp(k) = 1.0;
        Cd = -mdivide_left(Ad,multiply(Ad_tmp,mdivide_left(Ad,Bd)));
        EXPECT_NEAR(Cd(i,j),g[k],1.0E-12);
      }
    }
  }
}
