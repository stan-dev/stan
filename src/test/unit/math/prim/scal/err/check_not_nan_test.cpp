#include <stan/math/rev/arr/meta/var.hpp>
#include <stan/math/rev/arr/meta/var_stack.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <gtest/gtest.h>

using stan::math::check_not_nan;

TEST(ErrorHandlingScalar,CheckNotNan) {
  const char* function = "check_not_nan";
  double x = 0;

  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with finite x: " << x;

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with x = Inf: " << x;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with x = -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_not_nan(function, "x", x), std::domain_error) 
    << "check_not_nan should throw exception on NaN: " << x;
}


TEST(ErrorHandlingScalar,CheckNotNanVectorized) {
  int N = 5;
  const char* function = "check_not_nan";
  std::vector<double> x(N);

  x.assign(N, 0);
  EXPECT_TRUE(check_not_nan(function, "x", x)) 
    << "check_not_nan(vector) should be true with finite x: " << x[0];

  x.assign(N, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(check_not_nan(function, "x", x)) 
    << "check_not_nan(vector) should be true with x = Inf: " << x[0];

  x.assign(N, -std::numeric_limits<double>::infinity());
  EXPECT_TRUE(check_not_nan(function, "x", x)) 
    << "check_not_nan(vector) should be true with x = -Inf: " << x[0];

  x.assign(N, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_not_nan(function, "x", x), std::domain_error) 
    << "check_not_nan(vector) should throw exception on NaN: " << x[0];
}

TEST(ErrorHandlingScalar, CheckNotNanVectorized_one_indexed_message) {
  int N = 5;
  const char* function = "check_not_nan";
  std::vector<double> x(N);
  std::string message;

  x.assign(N, 0);
  x[2] = std::numeric_limits<double>::quiet_NaN();
  try {
    check_not_nan(function, "x", x);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[3]"))
    << message;
}

TEST(ErrorHandlingScalar, CheckNotNanVarCheckUnivariate) {
  using stan::agrad::var;

  const char* function = "check_not_nan";
  var a(5.0);

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_TRUE(1U == stack_size);
  EXPECT_TRUE(check_not_nan(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_TRUE(1U == stack_size_after_call);

  stan::agrad::recover_memory();
}

TEST(ErrorHandlingScalar, CheckNotNanVarCheckVectorized) {
  using stan::agrad::var;
  using std::vector;

  int N = 5;
  const char* function = "check_not_nan";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_TRUE(5U == stack_size);
  EXPECT_TRUE(check_not_nan(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_TRUE(5U == stack_size_after_call);
  stan::agrad::recover_memory();
}


