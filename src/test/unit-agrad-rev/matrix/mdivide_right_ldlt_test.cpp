#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/matrix/mdivide_right_ldlt.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/math/matrix/mdivide_right_spd.hpp>
#include <stan/agrad/rev/matrix/mdivide_left_spd.hpp>

TEST(AgradRevMatrix, mdivide_right_ldlt_vv) {
  using stan::agrad::var;
  using stan::agrad::row_vector_v;
  using stan::agrad::matrix_v;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_spd;
  using stan::math::mdivide_right_ldlt;
  using stan::math::LDLT_factor;  
  using std::vector;

  row_vector_v b(5);
  matrix_v A(5,5);
  row_vector_v x, x_basic;
  row_vector_d x_val, x_basic_val;
  row_vector_d expected(5);
  vector<var> vars;
  vector<double> grad, grad_basic;
  
  expected << 1, 2, 3, 4, 5;
  
  for (int i = 0; i < b.size(); i++) {
    // solve using mdivide_right_ldlt
    b << 62, 84, 84, 76, 108;
    A << 
      20, 8, -9,  7,  5, 
      8, 20,  0,  4,  4, 
     -9, 0,  20,  2,  5, 
      7, 4,  2,  20, -5, 
      5, 4,  5, -5,  20;
    LDLT_factor<var,-1,-1> ldlt_A;
    ldlt_A.compute(A);
    ASSERT_TRUE(ldlt_A.success());
    x = mdivide_right_ldlt(b, ldlt_A);
    x_val = stan::agrad::value_of(x);
    ASSERT_EQ(expected.size(), x_val.size());
    for (int n = 0; n < expected.size(); n++) {
      EXPECT_FLOAT_EQ(expected(n), x_val(n))
        << "value of mdivide_right_ldlt does not match"
        << " for element " << n;
    }

    vars.clear();
    for (int n = 0; n < b.size(); n++) {
      vars.push_back(b(n));
    }
    for (int n = 0; n < A.size(); n++) {
      vars.push_back(A(n));
    }
    x(i).grad(vars, grad);


    // solve using basic math
    b << 62, 84, 84, 76, 108;
    A << 
      20, 8, -9,  7,  5, 
      8, 20,  0,  4,  4, 
     -9, 0,  20,  2,  5, 
      7, 4,  2,  20, -5, 
      5, 4,  5, -5,  20;
    x_basic = mdivide_right_spd(b,A);
    x_basic_val = stan::agrad::value_of(x_basic);
    ASSERT_EQ(expected.size(), x_basic_val.size());
    for (int n = 0; n < expected.size(); n++) {
      EXPECT_FLOAT_EQ(expected(n), x_basic_val(n))
        << "value of basic math does not match"
        << " for element " << n;
    }

    vars.clear();
    for (int n = 0; n < b.size(); n++) {
      vars.push_back(b(n));
    }
    for (int n = 0; n < A.size(); n++) {
      vars.push_back(A(n));
    }
    x_basic(i).grad(vars, grad_basic);

    // test all gradients
    ASSERT_EQ(grad_basic.size(), grad.size());
    for (size_t n = 0; n < grad_basic.size(); n++)
      EXPECT_FLOAT_EQ(grad_basic[n], grad[n])
        << "for element " << i << ", gradient " << n
        << " does not match the basic auto-diff implementation";
  }
}


TEST(AgradRevMatrix, mdivide_right_ldlt_vd) {
  using stan::agrad::var;
  using stan::agrad::row_vector_v;
  using stan::math::matrix_d;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_ldlt;
  using stan::math::LDLT_factor;  
  using stan::math::mdivide_right_spd;
  using std::vector;

  row_vector_v b(5);
  matrix_d A(5,5);
  row_vector_v x, x_basic;
  row_vector_d x_val, x_basic_val;
  row_vector_d expected(5);
  vector<var> vars;
  vector<double> grad, grad_basic;
  
  expected << 1, 2, 3, 4, 5;
  
  for (int i = 0; i < b.size(); i++) {
    // solve using mdivide_right_ldlt
    b << 62, 84, 84, 76, 108;
    A << 
      20, 8, -9,  7,  5, 
      8, 20,  0,  4,  4, 
     -9, 0,  20,  2,  5, 
      7, 4,  2,  20, -5, 
      5, 4,  5, -5,  20;
    LDLT_factor<double,-1,-1> ldlt_A;
    ldlt_A.compute(A);
    ASSERT_TRUE(ldlt_A.success());
    x = mdivide_right_ldlt(b, ldlt_A);
    x_val = stan::agrad::value_of(x);
    ASSERT_EQ(expected.size(), x_val.size());
    for (int n = 0; n < expected.size(); n++) {
      EXPECT_FLOAT_EQ(expected(n), x_val(n))
        << "value of mdivide_right_ldlt does not match"
        << " for element " << n;
    }

    vars.clear();
    for (int n = 0; n < b.size(); n++) {
      vars.push_back(b(n));
    }
    x(i).grad(vars, grad);


    // solve using basic math
    b << 62, 84, 84, 76, 108;
    A << 
      20, 8, -9,  7,  5, 
      8, 20,  0,  4,  4, 
     -9, 0,  20,  2,  5, 
      7, 4,  2,  20, -5, 
      5, 4,  5, -5,  20;
    x_basic = mdivide_right_spd(b ,stan::agrad::to_var(A));
    x_basic_val = stan::agrad::value_of(x_basic);
    ASSERT_EQ(expected.size(), x_basic_val.size());
    for (int n = 0; n < expected.size(); n++) {
      EXPECT_FLOAT_EQ(expected(n), x_basic_val(n))
        << "value of basic math does not match"
        << " for element " << n;
    }

    vars.clear();
    for (int n = 0; n < b.size(); n++) {
      vars.push_back(b(n));
    }
    x_basic(i).grad(vars, grad_basic);

    // test all gradients
    ASSERT_EQ(grad_basic.size(), grad.size());
    for (size_t n = 0; n < grad_basic.size(); n++)
      EXPECT_FLOAT_EQ(grad_basic[n], grad[n])
        << "for element " << i << ", gradient " << n
        << " does not match the basic auto-diff implementation";
  }
}

TEST(AgradRevMatrix, mdivide_right_ldlt_dv) {
  using stan::agrad::var;
  using stan::agrad::row_vector_v;
  using stan::agrad::matrix_v;
  using stan::math::row_vector_d;
  using stan::math::mdivide_right_ldlt;
  using stan::math::LDLT_factor;  
  using stan::math::mdivide_right_spd;
  using std::vector;

  row_vector_d b(5);
  matrix_v A(5,5);
  row_vector_v x, x_basic;
  row_vector_d x_val, x_basic_val;
  row_vector_d expected(5);
  vector<var> vars;
  vector<double> grad, grad_basic;
  
  expected << 1, 2, 3, 4, 5;
  
  for (int i = 0; i < b.size(); i++) {
    // solve using mdivide_right_ldlt
    b << 62, 84, 84, 76, 108;
    A << 
      20, 8, -9,  7,  5, 
      8, 20,  0,  4,  4, 
     -9, 0,  20,  2,  5, 
      7, 4,  2,  20, -5, 
      5, 4,  5, -5,  20;
    LDLT_factor<var,-1,-1> ldlt_A;
    ldlt_A.compute(A);
    ASSERT_TRUE(ldlt_A.success());
    x = mdivide_right_ldlt(b, ldlt_A);
    x_val = stan::agrad::value_of(x);
    ASSERT_EQ(expected.size(), x_val.size());
    for (int n = 0; n < expected.size(); n++) {
      EXPECT_FLOAT_EQ(expected(n), x_val(n))
        << "value of mdivide_right_ldlt does not match"
        << " for element " << n;
    }

    vars.clear();
    for (int n = 0; n < A.size(); n++) {
      vars.push_back(A(n));
    }
    x(i).grad(vars, grad);


    // solve using basic math
    b << 62, 84, 84, 76, 108;
    A << 
      20, 8, -9,  7,  5, 
      8, 20,  0,  4,  4, 
     -9, 0,  20,  2,  5, 
      7, 4,  2,  20, -5, 
      5, 4,  5, -5,  20;
    x_basic = mdivide_right_spd(stan::agrad::to_var(b),A);
    x_basic_val = stan::agrad::value_of(x_basic);
    ASSERT_EQ(expected.size(), x_basic_val.size());
    for (int n = 0; n < expected.size(); n++) {
      EXPECT_FLOAT_EQ(expected(n), x_basic_val(n))
        << "value of basic math does not match"
        << " for element " << n;
    }

    vars.clear();
    for (int n = 0; n < A.size(); n++) {
      vars.push_back(A(n));
    }
    x_basic(i).grad(vars, grad_basic);

    // test all gradients
    ASSERT_EQ(grad_basic.size(), grad.size());
    for (size_t n = 0; n < grad_basic.size(); n++)
      EXPECT_FLOAT_EQ(grad_basic[n], grad[n])
        << "for element " << i << ", gradient " << n
        << " does not match the basic auto-diff implementation";
  }
}


