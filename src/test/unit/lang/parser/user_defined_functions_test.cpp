#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

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

TEST(parserFunctions, rejectfuns1) {
  test_parsable("lang/print_reject_function_gq");
}

TEST(parserFunctions, rejectfuns2) {
  test_parsable("lang/print_reject_function_model");
}

TEST(parserFunctions, rejectfuns3) {
  test_parsable("lang/print_reject_function_tdata");
}

TEST(parserFunctions, rejectfuns4) {
  test_parsable("lang/print_reject_function_tparams");
}

TEST(parserFunctions, rejectfuns5) {
  test_parsable("lang/reject_func_call_generated_quantities");
}

TEST(parserFunctions, rejectfuns6) {
  test_parsable("lang/reject_func_call_model");
}

TEST(parserFunctions, rejectfuns7) {
  test_parsable("lang/reject_func_call_transformed_data");
}

TEST(parserFunctions, rejectfuns8) {
  test_parsable("lang/reject_func_call_transformed_parameters");
}

TEST(parserFunctions, funsBad0) {
  test_throws("functions-bad0", "Functions cannot contain void argument types");
}

TEST(parserFunctions, funsBad1) {
  test_throws("functions-bad1", "Function already declared");
}

TEST(parserFunctions, funsBad2) {
  test_throws("functions-bad2", "Function declared, but not defined");
}

TEST(parserFunctions, funsBad2_good) {
  is_parsable("src/test/test-models/bad/functions-bad2.stan",0, true);
}

TEST(parserFunctions, funsBad3) {
  test_throws("functions-bad3", "SYNTAX ERROR, MESSAGE(S) FROM PARSER:");
}

TEST(parserFunctions,funsBad4) {
  test_throws("functions-bad4",
              "Functions used as statements must be declared to have void returns");
}

TEST(parserFunctions,funsBad5) {
  test_throws("functions-bad5",
              "Base type mismatch in assignment");
}

TEST(parserFunctions,funsBad6) {
  test_throws("functions-bad6",
              "Function target() or functions suffixed with _lp only"
              " allowed in transformed parameter block, model block");
}

TEST(parserFunctions,funsBad7) {
  test_throws("functions-bad7",
              "Function target() or functions suffixed with _lp only"
              " allowed in transformed parameter block, model block");
}

TEST(parserFunctions,funsBad11) {
  test_throws("functions-bad11",
              "Sampling statements (~) and increment_log_prob()");
}

TEST(parserFunctions,funsBad12) {
  test_throws("functions-bad12",
              "Sampling statements (~) and increment_log_prob()");
}

TEST(parserFunctions,funsBad13) {
  test_throws("functions-bad13",
              "Cannot assign to function argument variables.");
}

TEST(parserFunctions,funsBad14) {
  test_throws("functions-bad14",
              "Function already defined");
}

TEST(parserFunctions,funsBad15) {
  test_throws("functions-bad15",
              "Attempt to increment log prob with void expression");
}

TEST(parserFunctions,funsBad16) {
  test_throws("functions-bad16",
              "Function system defined");
}

TEST(parserFunctions,funsBad17) {
  test_throws("functions-bad17",
              "Real return type required for probability functions"
              " ending in _log, _lpdf, _lpmf, _lcdf, or _lccdf.");
}

TEST(parserFunctions, funsBad18) {
  test_throws("functions-bad18",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserFunctions, funsBad19) {
  test_throws("functions-bad19",
              "argument declared as real, defined as data real");
}

TEST(parserFunctions, funsBad20) {
  test_throws("functions-bad20",
              "argument declared as data real, defined as real");
}

TEST(parserFunctions, funsBad21) {
  test_throws("functions-bad21",
              "must be data only, found expression containing a parameter varaible");
}

TEST(parserFunctions, funsBadODE) {
  test_throws("functions-bad22-ode",
              "must be data only, found expression containing a parameter varaible");
}

TEST(parserFunctions, badProbFunSuffix) {
  test_throws("bad_prob_fun_suffix",
              "Probability function must end in _lpdf or _lpmf");
}

TEST(parserFunctions, voidFunReturn) {
  test_throws("functions-bad23",
              "Void returns only allowed from function bodies "
              "of void return type.");
}

TEST(parserFunctions, nonVoidFunReturn) {
  test_throws("functions-bad24",
              "Void function cannot return a value.");
}

TEST(parserFunctions, incompleteReturnStmt) {
  test_throws("functions-bad25",
              "Non-void function must return expression "
              "of specified return type.");
}


TEST(parserFunctions, returnNoSemi) {
  test_throws("functions-bad26",
              "PARSER EXPECTED: \";\"");
}

