#include <stan/math/matrix/inverse_spd.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/rev/print_stack.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/agrad/rev/functions/abs.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>

TEST(AgradRevMatrix,inverse_spd_val) {
  using stan::math::inverse_spd;
  using stan::agrad::matrix_v;

  matrix_v a(2,2);
  a << 2.0, 3.0, 
    3.0, 7.0;

  matrix_v a_inv = inverse_spd(a);

  matrix_v I = multiply(a,a_inv);

  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  EXPECT_THROW(inverse_spd(matrix_v(2,3)), std::domain_error);

  a << 2.0, 3.0, 
  1.0, 7.0;
  EXPECT_THROW(inverse_spd(a), std::domain_error);
  a << 1.0, -1.0, 
  -1.0, -1.0;
  EXPECT_THROW(inverse_spd(a), std::domain_error);
}

TEST(AgradRevMatrix,inverse_spd_grad) {
  using stan::math::inverse_spd;
  using stan::agrad::matrix_v;
  
  for (size_t k = 0; k < 2; ++k) {
    for (size_t l = 0; l < 2; ++l) {
      
      matrix_v ad(2,2);
      ad << 2.0, 3.0, 
      3.0, 7.0;
      
      AVEC x = createAVEC(ad(0,0),ad(0,1),ad(1,0),ad(1,1));
      
      matrix_v ad_inv = inverse_spd(ad);
      
      // int k = 0;
      // int l = 1;
      VEC g;
      (0.5*(ad_inv(k,l) + ad_inv(l,k))).grad(x,g);
      
      int idx = 0;
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          EXPECT_FLOAT_EQ(-0.5*ad_inv(k,i).val() * ad_inv(j,l).val()
                          -0.5*ad_inv(l,i).val() * ad_inv(j,k).val(), g[idx]);
          ++idx;
        }
      }
    }
  }
}

TEST(AgradRevMatrix,inverse_spd_inverse_spd_sum) {
  using stan::math::sum;
  using stan::math::inverse_spd;
  using stan::agrad::matrix_v;
  
  matrix_v a(4,4);
  a << 1.0, 0.0, 0.0, 0.0, 
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 1.0;
  
  AVEC x;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      x.push_back(a(i,j));
  
  AVAR a_inv_inv_sum = sum(inverse_spd(inverse_spd(a)));
  
  VEC g;
  a_inv_inv_sum.grad(x,g);

  size_t k = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(1.0,g[k]);
      k++;
    }
  }
}
