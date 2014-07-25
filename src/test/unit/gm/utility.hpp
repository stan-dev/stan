#ifndef TEST__UNIT__GM__PARSER__UTILITY_HPP
#define TEST__UNIT__GM__PARSER__UTILITY_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <exception>
#include <stdexcept>

#include <boost/lexical_cast.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/parser.hpp>
#include <stan/gm/generator.hpp>
#include <stan/gm/grammars/program_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>


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
 */
bool is_parsable(const std::string& file_name,
                 std::ostream* msgs = 0) {
  stan::gm::program prog;
  std::ifstream fs(file_name.c_str());
  std::string model_name = file_name_to_model_name(file_name);
  bool parsable = stan::gm::parse(msgs, fs, file_name, model_name, prog);
  return parsable;
}


/** test whether model with specified name in path syntax-only parses successfully
 *
 * @param model_name Name of model to parse
 * @param folder Path to folder under src/test/test-models (default "syntax-only")
 * @param msgs Warning message
 */
bool is_parsable_folder(const std::string& model_name,
                        const std::string folder = "syntax-only",
                        std::ostream* msgs = 0) {
  std::string path("src/test/test-models/");
  path += folder;
  path += "/";
  path += model_name;
  path += ".stan";
  return is_parsable(path,msgs);
}

/** test that model with specified name in folder "syntax-only"
 *  parses without throwing an exception
 *
 * @param model_name Name of model to parse
 */
void test_parsable(const std::string& model_name) {
  {
    SCOPED_TRACE("parsing: " + model_name);
    EXPECT_TRUE(is_parsable_folder(model_name, "syntax-only"));
  }
}


/** test that model with specified name in folder "reference" throws
 * an exception containing the second arg as a substring
 *
 * @param model_name Name of model to parse
 * @param msg Substring of error message expected.
 */
void test_throws(const std::string& model_name, const std::string& error_msg) {
  std::stringstream msgs;
  try {
    is_parsable_folder(model_name, "reference", &msgs);
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
         << " is parsable and were exepecting msg=" << error_msg
         << std::endl;
}

/** test that model with specified name in syntax-only path parses
 * and returns a warning containing the second arg as a substring
 *
 * @param model_name Name of model to parse
 * @param msg Substring of warning message expected.
 */
void test_warning(const std::string& model_name, const std::string& warning_msg) {
  std::stringstream msgs;
  EXPECT_TRUE(is_parsable_folder(model_name, "syntax-only", &msgs));
  EXPECT_TRUE(msgs.str().find_first_of(warning_msg) != std::string::npos);
}

#endif
