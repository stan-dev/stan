#include <gtest/gtest.h>
#include <exception>
#include <stdexcept>
#include <test/unit/lang/utility.hpp>

TEST(langParser, functionSigErrorsFunUnknown) {
  test_throws("signature_function_unknown",
              "Function foo_whatev_log not found");
}
TEST(langParser, functionSigErrorsFunKnown) {
  test_throws("signature_function_known",
              "No matches for:",
              "bernoulli_logit_log(vector, vector)",
              "Available argument signatures for bernoulli_logit_log:");
}
TEST(langParser, functionSigErrorsSampUnknown) {
  test_throws("signature_sampling_unknown",
              "Probability function must end in _lpdf or _lpmf."
              " Found distribution family = foo_whatev");
}
TEST(langParser, functionSigErrorsSampKnown) {
  test_throws("signature_sampling_known",
              "No matches for:",
              "vector ~ bernoulli_logit(vector)",
              "Available argument signatures for bernoulli_logit:");
}
TEST(langParser, functionSigErrorsMultiDef) {
  test_parsable("multiple_funs");
  test_throws("multi_fun",
              "Function already defined, name=foo");
}
TEST(langParser, functionSigErrorsFactorial) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/function-signatures/falling_factorial.stan"),
                           std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/function-signatures/rising_factorial.stan"),
                           std::invalid_argument);
}  
