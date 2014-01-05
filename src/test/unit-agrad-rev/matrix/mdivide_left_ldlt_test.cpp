#include <stan/agrad/rev/matrix/mdivide_left_ldlt.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/mdivide_left_ldlt.hpp>
#include <stan/math/matrix/mdivide_left_spd.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/math/matrix/typedefs.hpp>

std::vector<double> finite_differences(const size_t row, const size_t col,
                                         const stan::math::matrix_d A,
                                         const bool calc_A_partials,
                                         const stan::math::matrix_d B,
                                         const bool calc_B_partials) {
  const double e = 1e-8;
  std::vector<double> finite_diff;
  stan::math::matrix_d C_plus, C_minus;

  if (calc_A_partials) {
    for (size_type j = 0; j < A.cols(); j++) {
      for (size_type i = 0; i < A.rows(); i++) {
        stan::math::matrix_d A_plus(A), A_minus(A);
        A_plus(i,j) += e;
        A_plus(j,i) = A_plus(i,j);
        A_minus(i,j) -= e;
        A_minus(j,i) = A_minus(i,j);
        stan::math::LDLT_factor<double,-1,-1> ldlt_A_plus;
        stan::math::LDLT_factor<double,-1,-1> ldlt_A_minus;
        ldlt_A_plus.compute(A_plus);
        ldlt_A_minus.compute(A_minus);
      
        C_plus = stan::math::mdivide_left_ldlt(ldlt_A_plus, B);
        C_minus = stan::math::mdivide_left_ldlt(ldlt_A_minus, B);
        finite_diff.push_back((C_plus(row,col) - C_minus(row,col)) / (2*e));
      }
    }
  }
  if (calc_B_partials) {
    stan::math::LDLT_factor<double,-1,-1> ldlt_A;
    ldlt_A.compute(A);
    for (size_type j = 0; j < B.cols(); j++) {
      for (size_type i = 0; i < B.rows(); i++) {
        stan::math::matrix_d B_plus(B);
        stan::math::matrix_d B_minus(B);
        B_plus(i,j) += e;
        B_minus(i,j) -= e;
        
        C_plus = stan::math::mdivide_left_ldlt(ldlt_A, B_plus);
        C_minus = stan::math::mdivide_left_ldlt(ldlt_A, B_minus);
        finite_diff.push_back((C_plus(row,col) - C_minus(row,col)) / (2*e));
      }
    }
  }
  return finite_diff;
}


TEST(AgradRevMatrix,mdivide_left_ldlt_val) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_Av;
  matrix_v Av(2,2);
  matrix_d Ad(2,2);
  matrix_v I;

  Av << 2.0, 3.0, 
    3.0, 7.0;
  Ad << 2.0, 3.0, 
    3.0, 7.0;
  
  ldlt_Av.compute(Av);
  ASSERT_TRUE(ldlt_Av.success());
  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());

  I = mdivide_left_ldlt(ldlt_Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left_ldlt(ldlt_Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left_ldlt(ldlt_Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);
}

TEST(AgradRevMatrix,mdivide_left_ldlt_grad_vv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::multiply;
  using stan::math::mdivide_left_spd;

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
      stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_A;
      matrix_v A(2,2);
      matrix_v B(2,2);
      matrix_v C;

      for (k = 0; k < 4; k++) {
        A(k) = Ad(k);
        B(k) = Bd(k);
      }

      ldlt_A.compute(A);
      ASSERT_TRUE(ldlt_A.success());

      C = mdivide_left_ldlt(ldlt_A,B);
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

TEST(AgradRevMatrix,mdivide_left_ldlt_finite_diff_vv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::multiply;
  using stan::math::mdivide_left_spd;

  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  3.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;

  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  
  for (size_type i = 0; i < Bd.rows(); i++) {
    for (size_type j = 0; j < Bd.cols(); j++) {
      // compute derivatives
      matrix_v A(2,2), B(2,2), C;
      for (size_t k = 0; k < 4; k++) {
        A(k) = Ad(k);
        B(k) = Bd(k);
      }
      A(0,1) = A(1,0);
      stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_A;
      ldlt_A.compute(A);
      ASSERT_TRUE(ldlt_A.success());
      
      C = mdivide_left_ldlt(ldlt_A,B);
      AVEC x = createAVEC(A(0,0),A(0,1),A(0,1),A(1,1),
                          B(0,0),B(1,0),B(0,1),B(1,1));
      VEC gradient;
      C(i,j).grad(x,gradient);

      // compute finite differences
      VEC finite_diffs = finite_differences(i, j, Ad, true, Bd, true);
      
      ASSERT_EQ(gradient.size(), finite_diffs.size());
      for (size_t k = 0; k < gradient.size(); k++)
        EXPECT_NEAR(finite_diffs[k], gradient[k], 1e-4);
    }
  }
}

TEST(AgradRevMatrix,mdivide_left_ldlt_grad_dv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
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
      stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
      matrix_v B(2,2);
      matrix_v C;
      
      for (k = 0; k < 4; k++) {
        B(k) = Bd(k);
      }
      
      ldlt_Ad.compute(Ad);
      ASSERT_TRUE(ldlt_Ad.success());
      C = mdivide_left_ldlt(ldlt_Ad,B);
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

TEST(AgradRevMatrix,mdivide_left_ldlt_finite_diff_dv) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::multiply;
  using stan::math::mdivide_left_spd;

  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  3.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;

  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  
  for (size_type i = 0; i < Bd.rows(); i++) {
    for (size_type j = 0; j < Bd.cols(); j++) {
      // compute derivatives
      matrix_v B(2,2), C;
      for (size_t k = 0; k < 4; k++) {
        B(k) = Bd(k);
      }

      C = mdivide_left_ldlt(ldlt_Ad,B);
      AVEC x = createAVEC(B(0,0),B(1,0),B(0,1),B(1,1));
      VEC gradient;
      C(i,j).grad(x,gradient);

      // compute finite differences
      VEC finite_diffs = finite_differences(i, j, Ad, false, Bd, true);

      ASSERT_EQ(gradient.size(), finite_diffs.size());
      for (size_t k = 0; k < gradient.size(); k++)
        EXPECT_NEAR(finite_diffs[k], gradient[k], 1e-4);
    }
  }
}


TEST(AgradRevMatrix,mdivide_left_ldlt_grad_vd) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
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
      stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_A;
      matrix_v A(2,2);
      matrix_v C;
      
      for (k = 0; k < 4; k++) {
        A(k) = Ad(k);
      }
      
      ldlt_A.compute(A);
      ASSERT_TRUE(ldlt_A.success());
      C = mdivide_left_ldlt(ldlt_A,Bd);
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

TEST(AgradRevMatrix,mdivide_left_ldlt_finite_diff_vd) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::multiply;
  using stan::math::mdivide_left_spd;

  matrix_d Ad(2,2), Ad_tmp(2,2);
  matrix_d Bd(2,2), Bd_tmp(2,2);
  matrix_d Cd(2,2);
  
  Ad << 2.0, 3.0, 
  3.0, 7.0;
  Bd << 12.0, 13.0, 
  15.0, 17.0;

  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  ldlt_Ad.compute(Ad);
  ASSERT_TRUE(ldlt_Ad.success());
  
  for (size_type i = 0; i < Bd.rows(); i++) {
    for (size_type j = 0; j < Bd.cols(); j++) {
      // compute derivatives
      matrix_v A(2,2), C;
      for (size_t k = 0; k < 4; k++) {
        A(k) = Ad(k);
      }
      A(0,1) = A(1,0);
      stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_A;
      ldlt_A.compute(A);
      ASSERT_TRUE(ldlt_A.success());
      
      C = mdivide_left_ldlt(ldlt_A,Bd);
      AVEC x = createAVEC(A(0,0),A(0,1),A(0,1),A(1,1));
                          
      VEC gradient;
      C(i,j).grad(x,gradient);
      
      // compute finite differences
      VEC finite_diffs = finite_differences(i, j, Ad, true, Bd, false);

      ASSERT_EQ(gradient.size(), finite_diffs.size());
      for (size_t k = 0; k < gradient.size(); k++)
        EXPECT_NEAR(finite_diffs[k], gradient[k], 1e-4);
    }
  }
}

