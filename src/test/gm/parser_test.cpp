#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <exception>

#include "stan/gm/ast.hpp"
#include "stan/gm/parser.hpp"

bool is_parsable(const std::string& file_name) {
  stan::gm::program prog;
  std::ifstream fs(file_name.c_str());
  bool parsable = stan::gm::parse(fs, file_name, prog);
  return parsable;
}

TEST(gm_parser,eight_schools) {
  EXPECT_TRUE(is_parsable("src/models/eight_schools.stan"));
}

TEST(gm_parser,bugs_1_kidney) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/kidney/kidney.stan"));
}
TEST(gm_parser,bugs_1_leuk) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/leuk/leuk.stan"));
}
TEST(gm_parser,bugs_1_leukfr) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/leukfr/leukfr.stan"));
}
TEST(gm_parser,bugs_1_mice) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/mice/mice.stan"));
}
TEST(gm_parser,bugs_1_oxford) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/oxford/oxford.stan"));
}
TEST(gm_parser,bugs_1_rats) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/rats/rats.stan"));
}
TEST(gm_parser,bugs_1_salm) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/salm/salm.stan"));
}
TEST(gm_parser,bugs_1_seeds) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/seeds/seeds.stan"));
}
TEST(gm_parser,bugs_1_surgical) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/surgical/surgical.stan"));
}

TEST(gm_parser,bugs_2_beetles_cloglog) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/beetles/beetles_cloglog.stan"));
}
TEST(gm_parser,bugs_2_beetles_logit) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/beetles/beetles_logit.stan"));
}
TEST(gm_parser,bugs_2_beetles_probit) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/beetles/beetles_probit.stan"));
}
TEST(gm_parser,bugs_2_birats) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/birats/birats.stan"));
}
TEST(gm_parser,bugs_2_dugongs) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/dugongs/dugongs.stan"));
}
TEST(gm_parser,bugs_2_eyes) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/eyes/eyes.stan"));
}
TEST(gm_parser,bugs_2_ice) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/ice/ice.stan"));
}
TEST(gm_parser,bugs_2_stagnant) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/stagnant/stagnant.stan"));
}

TEST(gm_parser,good_trunc) {
  EXPECT_TRUE(is_parsable("src/test/gm/model_specs/good_trunc.stan"));
}


TEST(gm_parser,parsable_test_bad1) {
  EXPECT_THROW(is_parsable("src/test/gm/model_specs/bad1.stan"),
	       std::runtime_error);
}
TEST(gm_parser,parsable_test_bad2) {
  EXPECT_THROW(is_parsable("src/test/gm/model_specs/bad2.stan"),
	       std::runtime_error);
}

TEST(gm_parser,parsable_test_bad3) {
  EXPECT_THROW(is_parsable("src/test/gm/model_specs/bad3.stan"),
	       std::runtime_error);
}

TEST(gm_parser,parsable_test_bad4) {
  EXPECT_THROW(is_parsable("src/test/gm/model_specs/bad4.stan"),
	       std::runtime_error);
}
