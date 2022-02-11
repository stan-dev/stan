#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(TestUnitUtil, streams) {
  stan::test::capture_std_streams();
  std::cout << "foo";
  std::cerr << "bar";
  EXPECT_EQ("foo", stan::test::cout_ss.str());
  EXPECT_EQ("bar", stan::test::cerr_ss.str());

  stan::test::reset_std_streams();
  stan::test::capture_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
