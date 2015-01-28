#include <stan/error_handling/scalar/dom_err.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev/var.hpp>


const char* function_ = "function";
const char* y_name_ = "y";
const char* msg1_ = "error_message ";
const char* msg2_ = " after y";

class ErrorHandlingScalar_dom_err : public ::testing::Test {
public:
  void SetUp() {
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
      stan::error_handling::dom_err<T>
        (function_, y_name_, y, msg1_, msg2_);
      FAIL() << "expecting call to dom_err<> to throw a domain_error,"
             << "but threw nothing";
    } catch(std::domain_error& e) {
      EXPECT_EQ(expected_message_with_message(y), e.what());
    } catch(...) {
      FAIL() << "expecting call to dom_err<> to throw a domain_error,"
             << "but threw a different type";
    }

    try {
      stan::error_handling::dom_err<T>
        (function_, y_name_, y, msg1_);
      FAIL() << "expecting call to dom_err<> to throw a domain_error,"
             << "but threw nothing";
    } catch(std::domain_error& e) {
      EXPECT_EQ(expected_message_without_message(y), e.what());
    } catch(...) {
      FAIL() << "expecting call to dom_err<> to throw a domain_error,"
             << "but threw a different type";
    }

  }
};

TEST_F(ErrorHandlingScalar_dom_err, double) {
  double y = 10;
  
  test_throw<double>(y);
}

TEST_F(ErrorHandlingScalar_dom_err, var) {
  stan::agrad::var y = 10;
  
  test_throw<stan::agrad::var>(y);
}
