#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(parserFunctions, funsGood0) {
  test_parsable("validate_functions"); // tests proper definitions and use
}

TEST(parserFunctions, funsGood1) {
  test_parsable("functions-good1");
}

TEST(parserFunctions, funsGood2) {
  test_parsable("functions-good2");
}

TEST(parserFunctions, funsGood3) {
  test_parsable("functions-good3");
}

TEST(parserFunctions, funsGood4) {
  test_parsable("functions-good-void");
  test_parsable("functions-good-void"); // test twice to ensure
                                        // symbols are not saved
}

TEST(parserFunctions, funsBad18) {
  test_throws("functions-bad18","variable identifier (name) may not be reserved word");
}

TEST(parserFunctions, funsBad0) {
  test_throws("functions-bad0","Functions cannot contain void argument types");
}

TEST(parserFunctions, funsBad1) {
  test_throws("functions-bad1","Function already declared");
}

TEST(parserFunctions, funsBad2) {
  test_throws("functions-bad2","Function declared, but not defined");
}

TEST(parserFunctions, funsBad3) {
  test_throws("functions-bad3","SYNTAX ERROR, MESSAGE(S) FROM PARSER:");
}

TEST(parserFunctions,funsBad4) {
  test_throws("functions-bad4",
              "Functions used as statements must be declared to have void returns");
}

TEST(parserFunctions,funsBad5) {
  test_throws("functions-bad5",
              "base type mismatch in assignment");
}

TEST(parserFunctions,funsBad6) {
  test_throws("functions-bad6",
              "lp suffixed functions only allowed in");
}

TEST(parserFunctions,funsBad7) {
  test_throws("functions-bad7",
              "lp suffixed functions only allowed in");
}

TEST(parserFunctions,funsBad8) {
  test_throws("functions-bad8",
              "random number generators only allowed in");
}

TEST(parserFunctions,funsBad9) {
  test_throws("functions-bad9",
              "random number generators only allowed in");
}

TEST(parserFunctions,funsBad10) {
  test_throws("functions-bad10",
              "random number generators only allowed in");
}

TEST(parserFunctions,funsBad11) {
  test_throws("functions-bad11",
              "sampling only allowed in model");
}

TEST(parserFunctions,funsBad12) {
  test_throws("functions-bad12",
              "sampling only allowed in model");
}

TEST(parserFunctions,funsBad13) {
  test_throws("functions-bad13",
              "Illegal to assign to function argument variables");
}

TEST(parserFunctions,funsBad14) {
  test_throws("functions-bad14",
              "Function already defined");
}

TEST(parserFunctions,funsBad15) {
  test_throws("functions-bad15",
              "attempt to increment log prob with void expression");
}

TEST(parserFunctions,funsBad16) {
  test_throws("functions-bad16",
              "Function system defined");
}

TEST(parserFunctions,funsBad17) {
  test_throws("functions-bad17",
              "Require real return type for functions ending in _log");
}
