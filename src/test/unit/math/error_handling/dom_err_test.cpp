#include <stan/math/error_handling/dom_err.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev.hpp>

class MathErrorHandling_dom_err : public ::testing::Test {
public:
  void SetUp() {
    function_ = "function(%1%)";
    y_name_ = "y";
    error_message_ = " error_message %1% ";
  }


  template <class T, class T_msg>
  std::string expected_message(T y, T_msg msg) {
    std::stringstream expected_message;
    expected_message << "function("
                     << typeid(T).name()
                     << "): "
                     << y_name_
                     << " error_message "
                     << y
                     << " "
                     << msg;
    return expected_message.str();
  }


  template <class T, class T_result, class T_msg>
  void test_throw(T y, T_msg msg2) {
    try {
      stan::math::dom_err<T, T_result, T_msg>
        (function_, y, y_name_, error_message_, msg2, 0);
      FAIL() << "expecting call to dom_err<> to throw a domain_error,"
             << "but threw nothing";
    } catch(std::domain_error& e) {
      EXPECT_EQ(expected_message(y, msg2), e.what());
    } catch(...) {
      FAIL() << "expecting call to dom_err<> to throw a domain_error,"
             << "but threw a different type";
    }
  }

  const char* function_;
  const char* y_name_;
  const char* error_message_;
};

TEST_F(MathErrorHandling_dom_err, double_double_double) {
  typedef double T;
  typedef double T_result;
  typedef double T_msg;
  
  T y = 10;
  T_msg msg2 = 50;
  
  test_throw<T, T_result, T_msg>(y,msg2);
}

TEST_F(MathErrorHandling_dom_err, double_double_string) {
  typedef double T;
  typedef double T_result;
  typedef std::string T_msg;
  
  T y = 10;
  T_msg msg2 = "abcd";
  
  test_throw<T, T_result, T_msg>(y,msg2);
}

TEST_F(MathErrorHandling_dom_err, int_double_double) {
  typedef int T;
  typedef double T_result;
  typedef double T_msg;
  
  T y = 10;
  T_msg msg2 = 50;
  
  test_throw<T, T_result, T_msg>(y,msg2);
}


TEST_F(MathErrorHandling_dom_err, var_var_var) {
  typedef stan::agrad::var T;
  typedef stan::agrad::var T_result;
  typedef stan::agrad::var T_msg;
  
  T y = 10;
  T_msg msg2 = 50;
  
  test_throw<T, T_result, T_msg>(y,msg2);
}
