#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>

using stan::agrad::var;
using stan::error_handling::check_positive_finite;

TEST(AgradRevErrorHandlingScalar,CheckPositiveFinite) {
  const char* function = "check_positive_finite";
  var x = 1;
 
  EXPECT_TRUE(check_positive_finite(function, "x", x))
    << "check_positive_finite should be true with finite x: " << x;
  x = -1;
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on x= " << x;
  x = 0;
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on x= " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on Inf: " << x;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on NaN: " << x;
  stan::agrad::recover_memory();
}

// ---------- check_positive_finite: vector tests ----------
TEST(AgradRevErrorHandlingScalar,CheckPositiveFinite_Vector) {
  const char* function = "check_positive_finite";
  std::vector<var> x;
  
  x.clear();
  x.push_back (1.5);
  x.push_back (0.1);
  x.push_back (1);
  ASSERT_TRUE(check_positive_finite(function, "x", x)) 
    << "check_positive_finite should be true with finite x";

  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on Inf";

  x.clear();
  x.push_back(-1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on negative x";

  x.clear();
  x.push_back(0);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on x=0";

  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(-std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on -Inf";
  
  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
 << "check_positive_finite should throw exception on NaN";
  stan::agrad::recover_memory();
}

// ---------- check_positive_finite: matrix tests ----------
TEST(AgradRevErrorHandlingScalar,CheckPositiveFinite_Matrix) {
  const char* function = "check_positive_finite";
  Eigen::Matrix<var,Eigen::Dynamic,1> x;
  
  x.resize(3);
  x << 3, 2, 1;
  ASSERT_TRUE(check_positive_finite(function, "x", x))
    << "check_positive_finite should be true with finite x";

  x.resize(3);
  x << 2, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on Inf";

  x.resize(3);
  x << 0, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on x=0";

  x.resize(3);
  x << -1, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on x=-1";

  x.resize(3);
  x << 2, 1, -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on -Inf";
  
  x.resize(3);
  x << 1, 2, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on NaN";
  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckPositiveFiniteVarCheckUnivariate) {
  using stan::agrad::var;
  using stan::error_handling::check_positive_finite;

  const char* function = "check_positive_finite";
  var a(5.0);

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(check_positive_finite(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckPositiveFiniteVarCheckVectorized) {
  using stan::agrad::var;
  using std::vector;
  using stan::error_handling::check_positive_finite;

  int N = 5;
  const char* function = "check_positive_finite";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_THROW(check_positive_finite(function,"a",a),std::domain_error);
  EXPECT_TRUE(check_positive_finite(function,"a",a[2]));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);

  a[2] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function,"a",a),std::domain_error);
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(6U,stack_size_after_call);
  stan::agrad::recover_memory();
}
