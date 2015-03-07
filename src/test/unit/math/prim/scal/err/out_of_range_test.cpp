#include <stan/math/prim/scal/err/out_of_range.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

const char* function_ = "function";
const char* msg1_ = "error_message1 ";
const char* msg2_ = "error_message2 ";

class ErrorHandlingScalar_out_of_range : public ::testing::Test {
public:
  void SetUp() {
  }


  template <class T>
  std::string expected_message_with_0_messages(T y, size_t i) {
    std::stringstream expected_message;
    expected_message << "function: "
                     << "accessing element out of range. "
                     << "index " << i << " out of range; "
                     << "expecting index to be between "
                     << "1 and " << y.size();
    return expected_message.str();
  }

  template <class T>
  std::string expected_message_with_1_message(T y, size_t i) {
    std::stringstream expected_message;
    expected_message << expected_message_with_0_messages(y, i);
    expected_message << msg1_;
    return expected_message.str();
  }

  template <class T>
  std::string expected_message_with_2_messages(T y, size_t i) {
    std::stringstream expected_message;
    expected_message << expected_message_with_1_message(y, i);
    expected_message << msg2_;
    return expected_message.str();
  }



  template <class T>
  void test_throw(T y, size_t i) {
    using stan::math::out_of_range;
    
    EXPECT_THROW_MSG(out_of_range(function_, y.size(), i, msg1_, msg2_),
                     std::out_of_range,
                     expected_message_with_2_messages(y, i));

    EXPECT_THROW_MSG(out_of_range(function_, y.size(), i, msg1_),
                     std::out_of_range,
                     expected_message_with_1_message(y, i));

    EXPECT_THROW_MSG(out_of_range(function_, y.size(), i),
                     std::out_of_range,
                     expected_message_with_0_messages(y, i));
  }
};

TEST_F(ErrorHandlingScalar_out_of_range, double) {
  std::vector<double> y(4);
  
  test_throw(y, 0);
  test_throw(y, 5);
}
