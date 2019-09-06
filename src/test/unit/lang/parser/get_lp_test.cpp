#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, illegalScope) { 
  test_throws("get_lp_bad_scope1",
              "Function target() or functions suffixed with _lp only"
              " allowed in transformed parameter block, model block");
  test_throws("get_lp_bad_scope2",
              "Function target() or functions suffixed with _lp only"
              " allowed in transformed parameter block, model block");
}
TEST(langParser, legal) {
  test_parsable("get_lp_good");
}
TEST(langParser, withinLpFunction) {
  test_parsable("lp_in_fun");
}
TEST(langParser, getParamsUnconstrained) {
  test_parsable("unconstrained_params_var");
}
