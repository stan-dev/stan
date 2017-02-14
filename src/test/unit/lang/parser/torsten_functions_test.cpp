#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, PKModelOneCpt_function_signatures) {
    test_parsable("function-signatures/math/torsten/PKModelOneCpt");
}
TEST(lang_parser, PKModelTwoCpt_function_signatures) {
    test_parsable("function-signatures/math/torsten/PKModelTwoCpt");
}
TEST(lang_parser, linOdeModel_function_signatures) {
    test_parsable("function-signatures/math/torsten/linOdeModel");
}
TEST(lang_parser, generalCptModel_function_signatures) {
    test_parsable("function-signatures/math/torsten/generalCptModel");
}
