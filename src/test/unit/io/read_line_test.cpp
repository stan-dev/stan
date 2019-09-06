#include <stan/io/read_line.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

TEST(Io, ReadLine) {
  using stan::io::read_line;
  std::stringstream s;
  s << "";
  EXPECT_EQ("", read_line(s));
  EXPECT_EQ("", read_line(s));
}
TEST(Io, Readline2) {
  using stan::io::read_line;
  std::stringstream s;
  s << "foo bar\n";
  EXPECT_EQ("foo bar\n", read_line(s));
  EXPECT_EQ("", read_line(s));
}
TEST(Io, Readline3) {
  using stan::io::read_line;
  std::stringstream s;
  s << "foo bar\nbaz bing";
  EXPECT_EQ("foo bar\n", read_line(s));
  EXPECT_EQ("baz bing", read_line(s));
  EXPECT_EQ("", read_line(s));
}

