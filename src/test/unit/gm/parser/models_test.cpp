#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <test/unit/gm/utility.hpp>

TEST(gm_parser,eight_schools) {
  EXPECT_TRUE(is_parsable("src/models/misc/eight_schools/eight_schools.stan"));
}

TEST(gm_parser,bugs_1_kidney) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/kidney/kidney.stan"));
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

// why commented out?
//TEST(gm_parser,bugs_2_stagnant) {
//  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/stagnant/stagnant.stan"));
//  }

TEST(gm_parser,triangle_lp) {
  EXPECT_TRUE(is_parsable("src/models/basic_distributions/triangle.stan"));
}

