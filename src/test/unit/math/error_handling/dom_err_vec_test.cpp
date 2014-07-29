#include <stan/math/error_handling/dom_err_vec.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev.hpp>
#include <vector>

class MathErrorHandling_dom_err_vec : public ::testing::Test {
public:
  void SetUp() {
    function_ = "function(%1%)";
    y_name_ = "y";
    error_message_ = " error_message %1% ";
    index_ = 0;
  }

  template <class T, class T_msg>
  std::string expected_message(T y, T_msg msg) {
    std::stringstream expected_message;
    expected_message << "function("
                     << typeid(typename T::value_type).name()
                     << "): "
                     << y_name_
                     << "[" << 1 + index_ << "] "
                     << " error_message "
                     << y[index_]
                     << " "
                     << msg;
    return expected_message.str();
  }


  template <class T, class T_result, class T_msg>
  void test_throw(T y, T_msg msg2) {
    try {
      stan::math::dom_err_vec<T, T_result, T_msg>
        (index_, function_, y, y_name_, error_message_, msg2, 0);
      FAIL() << "expecting call to dom_err_vec<> to throw a domain_error,"
             << "but threw nothing";
    } catch(std::domain_error& e) {
      EXPECT_EQ(expected_message(y, msg2), e.what());
    } catch(...) {
      FAIL() << "expecting call to dom_err_vec<> to throw a domain_error,"
             << "but threw a different type";
    }
  }

  const char* function_;
  const char* y_name_;
  const char* error_message_;
  size_t index_;
};

TEST_F(MathErrorHandling_dom_err_vec, vdouble_double_double) {
  typedef std::vector<double> T;
  typedef double T_result;
  typedef double T_msg;
  
  T y;
  y.push_back(10);
  T_msg msg2 = 50;
  
  test_throw<T, T_result, T_msg>(y,msg2);
}

TEST_F(MathErrorHandling_dom_err_vec, vdouble_double_string) {
  typedef std::vector<double> T;
  typedef double T_result;
  typedef std::string T_msg;
  
  T y;
  y.push_back(10);
  T_msg msg2 = "abcd";
  
  test_throw<T, T_result, T_msg>(y,msg2);
}

TEST_F(MathErrorHandling_dom_err_vec, vint_double_double) {
  typedef std::vector<int> T;
  typedef double T_result;
  typedef double T_msg;
  
  T y;
  y.push_back(10);
  T_msg msg2 = 50;
  
  test_throw<T, T_result, T_msg>(y,msg2);
}


TEST_F(MathErrorHandling_dom_err_vec, vvar_var_var) {
  typedef std::vector<stan::agrad::var> T;
  typedef stan::agrad::var T_result;
  typedef double T_msg;
  
  T y;
  y.push_back(10);
  T_msg msg2 = 50;
  
  test_throw<T, T_result, T_msg>(y,msg2);
}

TEST_F(MathErrorHandling_dom_err_vec, one_indexed) {
  std::string message;
  int n = 5;
  std::vector<double> y(20);
  try {
    stan::math::dom_err_vec
      (n, function_, y, y_name_, error_message_, "", (double*)0);
    FAIL() << "expecting call to dom_err_vec<> to throw a domain_error,"
           << "but threw nothing";
  } catch(std::domain_error& e) {
    message = e.what();
  } catch(...) {
    FAIL() << "expecting call to dom_err_vec<> to throw a domain_error,"
           << "but threw a different type";
  }

  EXPECT_NE(std::string::npos, message.find("[6]"));
}
