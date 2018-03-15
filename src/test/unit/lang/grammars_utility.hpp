#ifndef TEST_UNIT_LANG_GRAMMARS_UTILITY_HPP
#define TEST_UNIT_LANG_GRAMMARS_UTILITY_HPP

#include <test/unit/new/grammars/test_functions_grammar_inst.cpp>
#include <test/unit/new/grammars/test_statement_grammar_inst.cpp>
#include <test/unit/new/grammars/test_block_var_decls_grammar_inst.cpp>
#include <test/unit/new/grammars/test_local_var_decls_grammar_inst.cpp>
#include <test/unit/new/grammars/test_bare_type_grammar_inst.cpp>
#include <stan/io/program_reader.hpp>

#include <stan/lang/ast_def.cpp>

// all of this needed for good error messages from parser
#include <stan/lang/generator/expression_visgen.hpp>
#include <stan/lang/generator/generate_array_builder_adds.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_idxs.hpp>
#include <stan/lang/generator/generate_idxs_user.hpp>
#include <stan/lang/generator/generate_idx.hpp>
#include <stan/lang/generator/generate_idx_user.hpp>
#include <stan/lang/generator/idx_visgen.hpp>
#include <stan/lang/generator/idx_user_visgen.hpp>

#include <stan/lang/parser.hpp>

#include <stan/lang/grammars/program_grammar_inst.cpp>
#include <stan/lang/grammars/functions_grammar_inst.cpp>
#include <stan/lang/grammars/statement_grammar_inst.cpp>
#include <stan/lang/grammars/statement_2_grammar_inst.cpp>
#include <stan/lang/grammars/block_var_decls_grammar_inst.cpp>
#include <stan/lang/grammars/local_var_decls_grammar_inst.cpp>
#include <stan/lang/grammars/bare_type_grammar_inst.cpp>
#include <stan/lang/grammars/common_adaptors_def.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>
#include <stan/lang/grammars/expression_grammar_inst.cpp>
#include <stan/lang/grammars/expression07_grammar_inst.cpp>
#include <stan/lang/grammars/term_grammar_inst.cpp>
#include <stan/lang/grammars/indexes_grammar_inst.cpp>
#include <stan/lang/grammars/whitespace_grammar_inst.cpp>
#include <stan/lang/grammars/semantic_actions_def.cpp>

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/spirit/home/support/iterators/line_pos_iterator.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <iostream>

// parse from file:  folder starts from below src/test/test-models
stan::lang::program
parse_program_from_file(const std::string& sub_folder,
                        const std::string& model_name,
                        bool& pass,
                        std::ostream& err_msgs) {
  pass = false;
  stan::lang::program parse_result;

  std::string file_name = model_name + ".stan";
  std::string path = "src/test/test-models/" + file_name;
  if (sub_folder.size() > 1)
    path = "src/test/test-models/" + sub_folder + "/" + file_name;
  std::ifstream fs(path.c_str());
  if (fs.fail()) {
    err_msgs <<  "Cannot open model file " << file_name << std::endl;
    std::cout << "Cannot open model file " << file_name << std::endl;
    return parse_result;
  }
  stan::io::program_reader reader;
  reader.add_event(0, 0, "start", "utility-stub.stan");
  reader.add_event(500, 500, "end", "utility-stub.stan");
  pass = stan::lang::parse(&err_msgs, fs, model_name + "_model",
                           reader, parse_result, false);
  return parse_result;
}

stan::lang::program
parse_program_from_file(const std::string& model_name,
                        bool& pass,
                        std::ostream& err_msgs) {
  return parse_program_from_file(std::string(), model_name, pass, err_msgs);
}

stan::lang::program
parse_program(std::string& input,
              bool& pass,
              std::ostream& err_msgs) {
  pass = false;

  std::istringstream istr(input);
  std::vector<std::string> search_path;
  search_path.push_back("foo");  
  std::stringstream ss(input);
  stan::io::program_reader reader(ss, "foo", search_path);
  stan::lang::program parse_result;
  pass = stan::lang::parse(&err_msgs, istr, "foo", reader, parse_result, false);
  return parse_result;
}

std::vector<stan::lang::function_decl_def>
parse_functions(std::string& input,
                bool& pass,
                std::ostream& err_msgs) {
  using boost::spirit::qi::expectation_failure;
  using boost::spirit::qi::phrase_parse;

  pass = false;

  //  std::cout << "parsing: " << std::endl << input << std::endl;
  std::vector<std::string> search_path;
  search_path.push_back("foo");  
  std::stringstream ss(input);
  stan::io::program_reader reader(ss, "foo", search_path);

  typedef std::string::const_iterator input_iterator;
  typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;

  lp_iterator fwd_begin = lp_iterator(input.begin());
  lp_iterator fwd_end = lp_iterator(input.end());

  // test_functions_grammar args:  reader, vm, msgs
  stan::lang::variable_map vm;
  std::stringstream msgs;

  // functions_grammar synthesis:  block_var_type
  std::vector<stan::lang::function_decl_def> parse_result;

  stan::lang::test_functions_grammar<lp_iterator> test_functions_grammar(reader, vm, msgs);
  stan::lang::whitespace_grammar<lp_iterator> whitesp_grammar(test_functions_grammar.error_msgs_);
  try {
    pass = phrase_parse(fwd_begin, fwd_end, test_functions_grammar,
                        whitesp_grammar, parse_result);
  } catch (const boost::spirit::qi::expectation_failure<lp_iterator>& e) {
    std::stringstream ss;
    ss << e.what_;
    std::string e_what = ss.str();
    std::string angle_eps("<eps>");
    if (e_what != angle_eps) {
      err_msgs << "PARSER EXPECTED: "
               << e.what_
               << std::endl;
    }
    if (test_functions_grammar.error_msgs_.str().length() != 0) {
      err_msgs << "SYNTAX ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_functions_grammar.error_msgs_.str()
               << std::endl;
    }
  } catch (const std::exception& e) {
      err_msgs << "PROGRAM ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_functions_grammar.error_msgs_.str()
               << std::endl;
  }
  if (fwd_begin != fwd_end) {
    pass = false;
    std::basic_stringstream<char> unparsed_non_ws;
    unparsed_non_ws << boost::make_iterator_range(fwd_begin, fwd_end);
    err_msgs << "PARSER FAILED TO PARSE INPUT"
             << std::endl
             << unparsed_non_ws.str()
             << std::endl;
  }
  return parse_result;
}


stan::lang::statement
parse_statement(std::string& input,
                bool& pass,
                std::ostream& err_msgs) {
  using boost::spirit::qi::expectation_failure;
  using boost::spirit::qi::phrase_parse;

  pass = false;

  //  std::cout << "parsing: " << std::endl << input << std::endl;
  std::vector<std::string> search_path;
  search_path.push_back("foo");  
  std::stringstream ss(input);
  stan::io::program_reader reader(ss, "foo", search_path);

  typedef std::string::const_iterator input_iterator;
  typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;

  lp_iterator fwd_begin = lp_iterator(input.begin());
  lp_iterator fwd_end = lp_iterator(input.end());

  // test_statement_grammar args:  reader, vm, msgs
  stan::lang::variable_map vm;
  std::stringstream msgs;

  // statement_grammar synthesis:  statement
  stan::lang::statement parse_result;

  stan::lang::test_statement_grammar<lp_iterator> test_statement_grammar(reader, vm, msgs);
  stan::lang::whitespace_grammar<lp_iterator> whitesp_grammar(test_statement_grammar.error_msgs_);
  try {
    pass = phrase_parse(fwd_begin, fwd_end, test_statement_grammar,
                        whitesp_grammar, parse_result);
  } catch (const boost::spirit::qi::expectation_failure<lp_iterator>& e) {
    std::stringstream ss;
    ss << e.what_;
    std::string e_what = ss.str();
    std::string angle_eps("<eps>");
    if (e_what != angle_eps) {
      err_msgs << "PARSER EXPECTED: "
               << e.what_
               << std::endl;
    }
    if (test_statement_grammar.error_msgs_.str().length() != 0) {
      err_msgs << "SYNTAX ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_statement_grammar.error_msgs_.str()
               << std::endl;
    }
  } catch (const std::exception& e) {
      err_msgs << "PROGRAM ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_statement_grammar.error_msgs_.str()
               << std::endl;
  }
  if (fwd_begin != fwd_end) {
    pass = false;
    std::basic_stringstream<char> unparsed_non_ws;
    unparsed_non_ws << boost::make_iterator_range(fwd_begin, fwd_end);
    err_msgs << "PARSER FAILED TO PARSE INPUT"
             << std::endl
             << unparsed_non_ws.str()
             << std::endl;
  }
  return parse_result;
}

std::vector<stan::lang::block_var_decl>
parse_var_decls(std::string& input,
                bool& pass,
                std::ostream& err_msgs) {
  using boost::spirit::qi::expectation_failure;
  using boost::spirit::qi::phrase_parse;

  pass = false;

  //  std::cout << "parsing: " << std::endl << input << std::endl;
  std::vector<std::string> search_path;
  search_path.push_back("foo");  
  std::stringstream ss(input);
  stan::io::program_reader reader(ss, "foo", search_path);

  typedef std::string::const_iterator input_iterator;
  typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;

  lp_iterator fwd_begin = lp_iterator(input.begin());
  lp_iterator fwd_end = lp_iterator(input.end());

  // test_block_var_decls_grammar args:  reader, vm, msgs
  stan::lang::variable_map vm;
  std::stringstream msgs;

  // block_var_decls_grammar synthesis:  block_var_type
  std::vector<stan::lang::block_var_decl> parse_result;

  stan::lang::test_block_var_decls_grammar<lp_iterator> test_block_var_decls_grammar(reader, vm, msgs);
  stan::lang::whitespace_grammar<lp_iterator> whitesp_grammar(test_block_var_decls_grammar.error_msgs_);
  try {
    pass = phrase_parse(fwd_begin, fwd_end, test_block_var_decls_grammar,
                        whitesp_grammar, parse_result);
  } catch (const boost::spirit::qi::expectation_failure<lp_iterator>& e) {
    std::stringstream ss;
    ss << e.what_;
    std::string e_what = ss.str();
    std::string angle_eps("<eps>");
    if (e_what != angle_eps) {
      err_msgs << "PARSER EXPECTED: "
               << e.what_
               << std::endl;
    }
    if (test_block_var_decls_grammar.error_msgs_.str().length() != 0) {
      err_msgs << "SYNTAX ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_block_var_decls_grammar.error_msgs_.str();
    }
  } catch (const std::exception& e) {
      err_msgs << "PROGRAM ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_block_var_decls_grammar.error_msgs_.str()
               << std::endl;
  }
  if (fwd_begin != fwd_end) {
    pass = false;
    std::basic_stringstream<char> unparsed_non_ws;
    unparsed_non_ws << boost::make_iterator_range(fwd_begin, fwd_end);
    err_msgs << "PARSER FAILED TO PARSE INPUT"
             << std::endl
             << unparsed_non_ws.str()
             << std::endl;
  }
  return parse_result;
}



std::vector<stan::lang::local_var_decl>
parse_local_var_decls(std::string& input,
                      bool& pass,
                      std::ostream& err_msgs) {
  using boost::spirit::qi::expectation_failure;
  using boost::spirit::qi::phrase_parse;

  pass = false;

  //  std::cout << "parsing: " << std::endl << input << std::endl;
  std::vector<std::string> search_path;
  search_path.push_back("foo");  
  std::stringstream ss(input);
  stan::io::program_reader reader(ss, "foo", search_path);

  typedef std::string::const_iterator input_iterator;
  typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;

  lp_iterator fwd_begin = lp_iterator(input.begin());
  lp_iterator fwd_end = lp_iterator(input.end());

  // test_local_var_decls_grammar args:  reader, vm, msgs
  stan::lang::variable_map vm;
  std::stringstream msgs;

  // local_var_decls_grammar synthesis:  local_var_type
  std::vector<stan::lang::local_var_decl> parse_result;

  stan::lang::test_local_var_decls_grammar<lp_iterator> test_local_var_decls_grammar(reader, vm, msgs);
  stan::lang::whitespace_grammar<lp_iterator> whitesp_grammar(test_local_var_decls_grammar.error_msgs_);
  try {
    pass = phrase_parse(fwd_begin, fwd_end, test_local_var_decls_grammar,
                        whitesp_grammar, parse_result);
  } catch (const boost::spirit::qi::expectation_failure<lp_iterator>& e) {
    std::stringstream ss;
    ss << e.what_;
    std::string e_what = ss.str();
    std::string angle_eps("<eps>");
    if (e_what != angle_eps) {
      err_msgs << "PARSER EXPECTED: "
               << e.what_
               << std::endl;
    }
    if (test_local_var_decls_grammar.error_msgs_.str().length() != 0) {
      err_msgs << "SYNTAX ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_local_var_decls_grammar.error_msgs_.str();
    }
  } catch (const std::exception& e) {
      err_msgs << "PROGRAM ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_local_var_decls_grammar.error_msgs_.str()
               << std::endl;
  }
  if (fwd_begin != fwd_end) {
    pass = false;
    std::basic_stringstream<char> unparsed_non_ws;
    unparsed_non_ws << boost::make_iterator_range(fwd_begin, fwd_end);
    err_msgs << "PARSER FAILED TO PARSE INPUT"
             << std::endl
             << unparsed_non_ws.str()
             << std::endl;
  }
  return parse_result;
}

stan::lang::bare_expr_type
parse_bare_type(std::string& input,
                bool& pass,
                std::ostream& err_msgs) {
  using boost::spirit::qi::expectation_failure;
  using boost::spirit::qi::phrase_parse;

  pass = false;

  //  std::cout << "parsing: " << std::endl << input << std::endl;
  std::vector<std::string> search_path;
  search_path.push_back("foo");  
  std::stringstream ss(input);
  stan::io::program_reader reader(ss, "foo", search_path);

  typedef std::string::const_iterator input_iterator;
  typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;

  lp_iterator fwd_begin = lp_iterator(input.begin());
  lp_iterator fwd_end = lp_iterator(input.end());

  // test_bare_type_grammar args:  reader, msgs
  std::stringstream msgs;

  // bare_type_grammar synthesis:  bare_expr_type
  stan::lang::bare_expr_type parse_result;

  stan::lang::test_bare_type_grammar<lp_iterator> test_bare_type_grammar(reader, msgs);
  stan::lang::whitespace_grammar<lp_iterator> whitesp_grammar(test_bare_type_grammar.error_msgs_);
  try {
    pass = phrase_parse(fwd_begin, fwd_end, test_bare_type_grammar,
                        whitesp_grammar, parse_result);
  } catch (const boost::spirit::qi::expectation_failure<lp_iterator>& e) {
    std::stringstream ss;
    ss << e.what_;
    std::string e_what = ss.str();
    std::string angle_eps("<eps>");
    if (e_what != angle_eps) {
      err_msgs << "PARSER EXPECTED: "
               << e.what_
               << std::endl;
    }
    if (test_bare_type_grammar.error_msgs_.str().length() != 0) {
      err_msgs << "SYNTAX ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_bare_type_grammar.error_msgs_.str();
    }
  } catch (const std::exception& e) {
      err_msgs << "PROGRAM ERROR, MESSAGE(S) FROM PARSER:"
               << std::endl
               << test_bare_type_grammar.error_msgs_.str()
               << std::endl;
  }
  if (fwd_begin != fwd_end) {
    pass = false;
    std::basic_stringstream<char> unparsed_non_ws;
    unparsed_non_ws << boost::make_iterator_range(fwd_begin, fwd_end);
    err_msgs << "PARSER FAILED TO PARSE INPUT"
             << std::endl
             << unparsed_non_ws.str()
             << std::endl;
  }
  return parse_result;
}


#endif
