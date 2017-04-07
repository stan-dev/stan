#ifndef TEST_UNIT_LANG_PARSER_UTILITY_HPP
#define TEST_UNIT_LANG_PARSER_UTILITY_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <exception>
#include <stdexcept>

#include <boost/lexical_cast.hpp>

#include <stan/lang/ast.hpp>
#include <stan/lang/compiler.hpp>
#include <stan/lang/parser.hpp>
#include <stan/lang/generator.hpp>
#include <stan/lang/grammars/program_grammar.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <stan/lang/grammars/expression_grammar.hpp>
#include <stan/lang/grammars/statement_grammar.hpp>
#include <stan/lang/grammars/var_decls_grammar.hpp>
#include <test/unit/util.hpp>

/** extract model name from filepath name
 * @param file_name  Name off model file
 */
std::string file_name_to_model_name(const std::string& name) {
  std::string name_copy = name;
  size_t last_bk = name_copy.find_last_of('\\');
  if (last_bk != std::string::npos)
    name_copy.erase(0,last_bk + 1);
  size_t last_fwd = name_copy.find_last_of('/');
  if (last_fwd != std::string::npos)
    name_copy.erase(0,last_fwd + 1);
    
  size_t last_dot = name_copy.find_last_of('.');
  if (last_dot != std::string::npos)
    name_copy.erase(last_dot,name_copy.size());

  name_copy += "_model";
  return name_copy;
}


/** test whether model with specified path name parses successfully
 *
 * @param file_name  Filepath of model file
 * @param msgs Expected error message (default: none)
 * @param allow_undefined Boolean to permit undefined functions (default: false)
 */
bool is_parsable(const std::string& file_name,
                 std::ostream* msgs = 0,
                 bool allow_undefined = false) {
  stan::lang::program prog;
  std::ifstream fs(file_name.c_str());
  std::string model_name = file_name_to_model_name(file_name);
  bool parsable
    = stan::lang::parse(msgs, fs, model_name, prog, allow_undefined);
  return parsable;
}


/** test whether model with specified name in path good parses successfully
 *
 * @param model_name Name of model to parse
 * @param folder Path to folder under src/test/test-models (default "good")
 * @param msgs Warning message
 */
bool is_parsable_folder(const std::string& model_name,
                        const std::string folder = "good",
                        std::ostream* msgs = 0) {
  std::string path("src/test/test-models/");
  path += folder;
  path += "/";
  path += model_name;
  path += ".stan";
  return is_parsable(path, msgs, false);
}

/** test that model with specified name in folder "good"
 *  parses without throwing an exception
 *
 * @param model_name Name of model to parse
 */
void test_parsable(const std::string& model_name) {
  {
    SCOPED_TRACE("parsing: " + model_name);
    EXPECT_TRUE(is_parsable_folder(model_name, "good"));
  }
}

/** test that model with specified name in folder "bad" throws
 * an exception containing the second arg as a substring
 *
 * @param model_name Name of model to parse
 * @param msg Substring of error message expected.
 */
void test_throws(const std::string& model_name, const std::string& error_msg) {
  std::stringstream msgs;
  try {
    is_parsable_folder(model_name, "bad", &msgs);
  } catch (const std::invalid_argument& e) {
    if (std::string(e.what()).find(error_msg) == std::string::npos
        && msgs.str().find(error_msg) == std::string::npos) {
      FAIL() << std::endl << "*********************************" << std::endl
             << "model name=" << model_name << std::endl
             << "*** EXPECTED: error_msg=" << error_msg << std::endl
             << "*** FOUND: e.what()=" << e.what() << std::endl
             << "*** FOUND: msgs.str()=" << msgs.str() << std::endl
             << "*********************************" << std::endl
             << std::endl;
    }
    return;
  }
  
  FAIL() << "model name=" << model_name 
         << " is parsable and were expecting msg=" << error_msg
         << std::endl;
}

/**
 * Same as test_throws() but for two messages.
 */
void test_throws(const std::string& model_name, 
                 const std::string& error_msg1,
                 const std::string& error_msg2) {
  test_throws(model_name, error_msg1);
  test_throws(model_name, error_msg2);
}

/**
 * Same as test_throws() but for three messages.
 */
void test_throws(const std::string& model_name, 
                 const std::string& error_msg1,
                 const std::string& error_msg2,
                 const std::string& error_msg3) {
  test_throws(model_name, error_msg1);
  test_throws(model_name, error_msg2);
  test_throws(model_name, error_msg3);
}

/** test that model with specified name in good path parses
 * and returns a warning containing the second arg as a substring
 *
 * @param model_name Name of model to parse
 * @param msg Substring of warning message expected.
 */
void test_warning(const std::string& model_name, 
                  const std::string& warning_msg) {
  std::stringstream msgs;
  EXPECT_TRUE(is_parsable_folder(model_name, "good", &msgs));
  bool found = msgs.str().find(warning_msg) != std::string::npos;
  EXPECT_TRUE(found) << std::endl 
    << "FOUND: " << msgs.str() 
    << std::endl
    << "EXPECTED (as substring): " << warning_msg
    << std::endl;
}

std::string model_to_cpp(const std::string& model_text) {
  std::string model_name = "foo";
  std::stringstream ss(model_text);
  std::stringstream msgs;
  stan::lang::program prog;
  bool parsable = stan::lang::parse(&msgs, ss, model_name, prog);
  EXPECT_TRUE(parsable);

  std::stringstream output;
  stan::lang::generate_cpp(prog, model_name, output);
  return output.str();
}


void expect_matches(int n,
                    const std::string& stan_code,
                    const std::string& target) {
  std::string model_cpp = model_to_cpp(stan_code);
  EXPECT_EQ(n, count_matches(target, model_cpp))
    << "looking for: " << target;
}

std::string get_file_name(const std::string& folder,
                          const std::string& model_name) {
  std::string path("src/test/test-models/");
  path += folder;
  path += "/";
  path += model_name;
  path += ".stan";
  return path;
}

void expect_match(const std::string& model_name,
                  const std::string& target,
                  bool allow_undefined = false) {
  std::stringstream msgs;
  std::string file_name = get_file_name("good", model_name);
  std::ifstream file_stream(file_name.c_str());
  std::stringstream cpp_out_stream;
  stan::lang::compile(&msgs, file_stream, cpp_out_stream,
                      model_name, allow_undefined);
  std::string cpp_out = cpp_out_stream.str();
  file_stream.close();
  EXPECT_TRUE(count_matches(target, cpp_out) > 0)
    << "looking for: " << target;
}

/**
 * Thest that model of specified name in test/test-models/good
 * has exactly the specified number of matches 
 *
 * @param[in] model_name Name of model file.
 * @param[in] warning_msg Message to count.
 * @param[in] n Expected number of message occurrences.
 */
void test_num_warnings(const std::string& model_name, 
                       const std::string& warning_msg,
                       int n) {
  std::stringstream msgs;
  EXPECT_TRUE(is_parsable_folder(model_name, "good", &msgs));
  EXPECT_EQ(n, count_matches(warning_msg, msgs.str()))
    << "looking for: " << warning_msg;
}
#endif

