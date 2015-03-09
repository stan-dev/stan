#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/err/invalid_argument_vec.hpp>
#include <vector>
#include <gtest/gtest.h>

const char* function_ = "function";
const char* y_name_ = "y";
const char* msg1_ = "error_message ";
const char* msg2_ = " second message";

class ErrorHandlingScalar_invalid_argument_vec : public ::testing::Test {
public:
  void SetUp() {
    index_ = 0;
  }

  template <class T>
  std::string expected_message_with_message(T y) {
    using stan::math::value_type;
    std::stringstream expected_message;
    expected_message << "function: "
                     << y_name_
                     << "[" << 1 + index_ << "] "
                     << "error_message "
                     << y[index_]
                     << " second message";
    return expected_message.str();
  }

  template <class T>
  std::string expected_message_without_message(T y) {
    using stan::math::value_type;
    std::stringstream expected_message;
    expected_message << "function: "
                     << y_name_
                     << "[" << 1 + index_ << "] "
                     << "error_message "
                     << y[index_];
    return expected_message.str();
  }


  template <class T>
  void test_throw(T y) {
    try {
      stan::math::invalid_argument_vec<T>
        (function_, y_name_, y, index_, msg1_, msg2_);
      FAIL() << "expecting call to invalid_argument_vec<> to throw a invalid_argument,"
             << "but threw nothing";
    } catch(std::invalid_argument& e) {
      EXPECT_EQ(expected_message_with_message(y), e.what());
    } catch(...) {
      FAIL() << "expecting call to invalid_argument_vec<> to throw a invalid_argument,"
             << "but threw a different type";
    }

    try {
      stan::math::invalid_argument_vec<T>
        (function_, y_name_, y, index_, msg1_);
      FAIL() << "expecting call to invalid_argument_vec<> to throw a invalid_argument,"
             << "but threw nothing";
    } catch(std::invalid_argument& e) {
      EXPECT_EQ(expected_message_without_message(y), e.what());
    } catch(...) {
      FAIL() << "expecting call to invalid_argument_vec<> to throw a invalid_argument,"
             << "but threw a different type";
    }
  }

  size_t index_;
};

TEST_F(ErrorHandlingScalar_invalid_argument_vec, vdouble) {
  std::vector<double> y;
  y.push_back(10);
  
  test_throw<std::vector<double> >(y);
}

TEST_F(ErrorHandlingScalar_invalid_argument_vec, vint) {
  std::vector<int> y;
  y.push_back(10);
  
  test_throw<std::vector<int> >(y);
}

TEST_F(ErrorHandlingScalar_invalid_argument_vec, one_indexed) {
  std::string message;
  int n = 5;
  std::vector<double> y(20);
  try {
    stan::math::invalid_argument_vec
      (function_, y_name_, y, n, msg1_, msg2_);
    FAIL() << "expecting call to invalid_argument_vec<> to throw a invalid_argument,"
           << "but threw nothing";
  } catch(std::invalid_argument& e) {
    message = e.what();
  } catch(...) {
    FAIL() << "expecting call to invalid_argument_vec<> to throw a invalid_argument,"
           << "but threw a different type";
  }

  EXPECT_NE(std::string::npos, message.find("[6]"));
}
