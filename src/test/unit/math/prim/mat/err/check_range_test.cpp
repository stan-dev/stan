#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_range.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(ErrorHandlingMatrix, checkRange_6_arg_std_vector) {
    using stan::math::check_range;
    std::vector<double> x;

    x.resize(4);
  
    EXPECT_TRUE(check_range("function", "x", 4, 1, 4, ""));
    EXPECT_TRUE(check_range("function", "x", 4, 2, 4, ""));
    EXPECT_TRUE(check_range("function", "x", 4, 3, 4, ""));
    EXPECT_TRUE(check_range("function", "x", 4, 4, 4, ""));


    std::string expected_message;
    expected_message = 
      "function: accessing element out of range. "
      "index 12 out of range; "
      "expecting index to be between 1 and 4; "
      "index position = 4";
    EXPECT_THROW_MSG(check_range("function", "x", 4, 12, 4, ""),
                     std::out_of_range,
                     expected_message);
    
}

TEST(ErrorHandlingMatrix, checkRange_4_arg_std_vector) {
    using stan::math::check_range;
    std::vector<double> x;

    x.resize(4);
  
    EXPECT_TRUE(check_range("function", "x", 4, 1));
    EXPECT_TRUE(check_range("function", "x", 4, 2));
    EXPECT_TRUE(check_range("function", "x", 4, 3));
    EXPECT_TRUE(check_range("function", "x", 4, 4));


    std::string expected_message;
    expected_message = 
      "function: accessing element out of range. "
      "index 12 out of range; "
      "expecting index to be between 1 and 4";
    EXPECT_THROW_MSG(check_range("function", "x", 4, 12),
                     std::out_of_range,
                     expected_message);
}
