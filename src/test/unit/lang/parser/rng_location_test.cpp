#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

void verify_bad_rng(const std::string& model_name) {
  test_throws(model_name,
              "Random number generators only allowed in"
              " transformed data block, generated quantities block"
              " or user-defined functions with names ending in _rng");
}

TEST(lang_parser, good_rng_location) {
  test_parsable("rng_loc");
}
TEST(lang_parser, bad_rng_location) {
  verify_bad_rng("rng_loc1");
  verify_bad_rng("rng_loc2");
  verify_bad_rng("rng_loc3");
  verify_bad_rng("rng_loc4");
  verify_bad_rng("rng_loc5");
  verify_bad_rng("rng_loc6");
}

