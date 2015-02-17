#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/subtract.hpp>
#include <stan/math/prim/mat/fun/get_base1.hpp>
#include <stan/math/prim/mat/fun/dot_product.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/assign.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/core/operator_addition.hpp>
#include <stan/math/rev/core/operator_divide_equal.hpp>
#include <stan/math/rev/core/operator_division.hpp>
#include <stan/math/rev/core/operator_equal.hpp>
#include <stan/math/rev/core/operator_greater_than.hpp>
#include <stan/math/rev/core/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/core/operator_less_than.hpp>
#include <stan/math/rev/core/operator_less_than_or_equal.hpp>
#include <stan/math/rev/core/operator_minus_equal.hpp>
#include <stan/math/rev/core/operator_multiplication.hpp>
#include <stan/math/rev/core/operator_multiply_equal.hpp>
#include <stan/math/rev/core/operator_not_equal.hpp>
#include <stan/math/rev/core/operator_plus_equal.hpp>
#include <stan/math/rev/core/operator_subtraction.hpp>
#include <stan/math/rev/core/operator_unary_decrement.hpp>
#include <stan/math/rev/core/operator_unary_increment.hpp>
#include <stan/math/rev/core/operator_unary_negative.hpp>
#include <stan/math/rev/core/operator_unary_not.hpp>
#include <stan/math/rev/core/operator_unary_plus.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/rev/mat/fun/dot_product.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

TEST(AgradRevMatrix,mv_squaredNorm) {
  using stan::agrad::matrix_v;

  matrix_v a(2,2);
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
TEST(AgradRevMatrix,mv_norm) {
  using stan::agrad::matrix_v;

  matrix_v a(2,1);
  a << -3.0, 4.0;
  
  AVEC x = createAVEC(a(0,0), a(1,0));

  AVAR s = a.norm();
  EXPECT_FLOAT_EQ(5.0,s.val());

  // (see hypot in special_functions_test) 
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(-3.0/5.0, g[0]);
  EXPECT_FLOAT_EQ(4.0/5.0, g[1]);
}  
TEST(AgradRevMatrix,mv_lp_norm) {
  using stan::agrad::matrix_v;

  matrix_v a(2,2);
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
TEST(AgradRevMatrix,mv_lp_norm_inf) {
  using stan::agrad::matrix_v;

  matrix_v a(2,2);
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
