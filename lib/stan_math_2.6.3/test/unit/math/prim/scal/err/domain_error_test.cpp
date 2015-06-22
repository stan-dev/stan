#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <gtest/gtest.h>
#include <sstream>

const char* function_ = "function";
const char* y_name_ = "y";
const char* msg1_ = "error_message ";
const char* msg2_ = " after y";

class ErrorHandlingScalar_domain_error : public ::testing::Test {
public:
  void SetUp() {
  }


  template <class T>
  std::string expected_message_with_message(T y) {
    std::stringstream expected_message;
    expected_message << "function: "
                     << y_name_
                     << " error_message "
                     << y
                     << " after y";
    return expected_message.str();
  }

  template <class T>
  std::string expected_message_without_message(T y) {
    std::stringstream expected_message;
    expected_message << "function: "
                     << y_name_
                     << " error_message "
                     << y;
    return expected_message.str();
  }


  template <class T>
  void test_throw(T y) {
    try {
      stan::math::domain_error<T>
        (function_, y_name_, y, msg1_, msg2_);
      FAIL() << "expecting call to domain_error<> to throw a domain_error,"
             << "but threw nothing";
    } catch(std::domain_error& e) {
      EXPECT_EQ(expected_message_with_message(y), e.what());
    } catch(...) {
      FAIL() << "expecting call to domain_error<> to throw a domain_error,"
             << "but threw a different type";
    }

    try {
      stan::math::domain_error<T>
        (function_, y_name_, y, msg1_);
      FAIL() << "expecting call to domain_error<> to throw a domain_error,"
             << "but threw nothing";
    } catch(std::domain_error& e) {
      EXPECT_EQ(expected_message_without_message(y), e.what());
    } catch(...) {
      FAIL() << "expecting call to domain_error<> to throw a domain_error,"
             << "but threw a different type";
    }

  }
};

TEST_F(ErrorHandlingScalar_domain_error, double) {
  double y = 10;
  
  test_throw<double>(y);
}
