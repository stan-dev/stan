#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, functionSigErrorsFunUnknown) { 
  test_throws("signature_function_unknown",
              "No matches for:",
              "  foo_whatev_log(vector, vector)",
              "Function foo_whatev_log not found.");
}
TEST(langParser, functionSigErrorsFunKnown) { 
  test_throws("signature_function_known",
              "No matches for:",
              "bernoulli_logit_log(vector, vector)",
              "Available argument signatures for bernoulli_logit_log:");
}
TEST(langParser, functionSigErrorsSampUnknown) { 
  test_throws("signature_sampling_unknown",
              "Distribution foo_whatev not found."
              " Require function with _lpdf or _lpmf or _log suffix");
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
