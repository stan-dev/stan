#include <stan/error_handling/invalid_argument.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev/var.hpp>

class ErrorHandlingScalar_invalid_argument : public ::testing::Test {
public:
  void SetUp() {
    function_ = "function";
    y_name_ = "y";
    msg1_ = "error_message ";
    msg2_ = " after y";
  }


  template <class T>
  std::string expected_message_with_message(T y) {
    std::stringstream expected_message;
    expected_message << "function("
                     << typeid(T).name()
                     << "): "
                     << y_name_
                     << " error_message "
                     << y
                     << " after y";
    return expected_message.str();
  }

  template <class T>
  std::string expected_message_without_message(T y) {
    std::stringstream expected_message;
    expected_message << "function("
                     << typeid(T).name()
                     << "): "
                     << y_name_
                     << " error_message "
                     << y;
    return expected_message.str();
  }


  template <class T>
  void test_throw(T y) {
    try {
      stan::math::invalid_argument<T>
        (function_, y_name_, y, msg1_, msg2_);
      FAIL() << "expecting call to invalid_argument<> to throw a invalid_argument,"
             << "but threw nothing";
    } catch(std::invalid_argument& e) {
      EXPECT_EQ(expected_message_with_message(y), e.what());
    } catch(...) {
      FAIL() << "expecting call to invalid_argument<> to throw a invalid_argument,"
             << "but threw a different type";
    }

    try {
      stan::math::invalid_argument<T>
        (function_, y_name_, y, msg1_);
      FAIL() << "expecting call to invalid_argument<> to throw a invalid_argument,"
             << "but threw nothing";
    } catch(std::invalid_argument& e) {
      EXPECT_EQ(expected_message_without_message(y), e.what());
    } catch(...) {
      FAIL() << "expecting call to invalid_argument<> to throw a invalid_argument,"
             << "but threw a different type";
    }

  }

  std::string function_;
  std::string y_name_;
  std::string msg1_;
  std::string msg2_;
};

TEST_F(ErrorHandlingScalar_invalid_argument, double) {
  double y = 10;
  
  test_throw<double>(y);
}

TEST_F(ErrorHandlingScalar_invalid_argument, var) {
  stan::agrad::var y = 10;
  
  test_throw<stan::agrad::var>(y);
}
