#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, funsNewLSyntax) {
  test_parsable("fun-defs-lpdf");
}
TEST(langParser, deprecateOldLogFuns) {
  test_warning("old-log-funs",
               "Warning: Function name 'multiply_log' is deprecated"
               " and will be removed in a later release; please"
               " replace with 'lmultiply'");
  test_warning("old-log-funs",
               "Warning: Function name 'binomial_coefficient_log' is"
               " deprecated and will be removed in a later release;"
               " please replace with 'lchoose'");
}
TEST(langParser, deprecateOldProbLogFuns) {
  test_warning("deprecate-old-prob-funs",
               "Warning: Deprecated function 'normal_log'; please replace"
               " suffix '_log' with '_lpdf' for density functions"
               " or '_lpmf' for mass functions");
  test_warning("deprecate-old-prob-funs",
               "Warning: Deprecated function 'normal_cdf_log'; please"
               " replace suffix '_cdf_log' with '_lcdf'");
  test_warning("deprecate-old-prob-funs",
               "Warning: Deprecated function 'normal_ccdf_log'; please"
               " replace suffix '_ccdf_log' with '_lccdf'");
}
TEST(langParser, newProbFunSuffixes) {
  test_parsable("new-prob-fun-suffixes");
}
TEST(langParser, userTruncation) {
  test_parsable("user-distro-truncate");
}
