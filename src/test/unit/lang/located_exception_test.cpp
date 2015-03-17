#include <stdexcept>
#include <stan/lang/located_exception.hpp>
#include <gtest/gtest.h>

TEST(langLocatedException, what) {
  using std::domain_error;
  using std::string;
  using stan::lang::located_exception;

  // makes sure nested error and line number are included in what() message
  domain_error de("foo");
  located_exception le(de, 3);
  string de_what = de.what();
  string le_what = le.what();

  EXPECT_TRUE(le_what.find_first_of(de_what) != string::npos);
  EXPECT_TRUE(le_what.find_first_of("3") != string::npos);

  located_exception lle(le, 5);
  string lle_what = lle.what();
  EXPECT_TRUE(lle_what.find_first_of(de_what) != string::npos);
  EXPECT_TRUE(lle_what.find_first_of(le_what) != string::npos);
  EXPECT_TRUE(lle_what.find_first_of("3") != string::npos);
  EXPECT_TRUE(lle_what.find_first_of("5") != string::npos);
}
TEST(langLocatedException, nestedException) {
  using std::domain_error;
  using std::string;
  using stan::lang::located_exception;

  domain_error de("foo");
  located_exception le(de, 3);
  EXPECT_EQ(&de, &le.nested_exception());
}
TEST(langLocatedException, line) {
 using std::domain_error;
  using std::string;
  using stan::lang::located_exception;

  domain_error de("foo");
  located_exception le(de, 3);
  EXPECT_EQ(3, le.line());
}
TEST(langLocatedException, baseException) {
  using std::domain_error;
  using std::string;
  using stan::lang::located_exception;

  domain_error de("foo");
  located_exception le(de, 3);
  EXPECT_EQ(&de, &le.base_exception());

  located_exception lle(le, 10);
  EXPECT_EQ(&de, &lle.base_exception());
}
TEST(langLocatedException, baseExceptionIs) {
  using std::domain_error;
  using std::invalid_argument;
  using std::string;
  using stan::lang::located_exception;

  domain_error de("foo");
  located_exception le(de, 3);
  EXPECT_TRUE(le.base_exception_is<domain_error>());
  EXPECT_FALSE(le.base_exception_is<invalid_argument>());
  EXPECT_FALSE(le.base_exception_is<located_exception>());
  
  located_exception lle(le, 10);
  EXPECT_TRUE(lle.base_exception_is<domain_error>());
  EXPECT_FALSE(lle.base_exception_is<invalid_argument>());
  EXPECT_FALSE(lle.base_exception_is<located_exception>());
  
}




