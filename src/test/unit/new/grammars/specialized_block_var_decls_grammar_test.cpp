#include <stan/io/program_reader.hpp>

#include <stan/lang/ast_def.cpp>

#include <stan/lang/grammars/block_var_decls_grammar_inst.cpp>
#include <stan/lang/grammars/common_adaptors_def.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>
#include <stan/lang/grammars/expression_grammar_inst.cpp>
#include <stan/lang/grammars/expression07_grammar_inst.cpp>
#include <stan/lang/grammars/term_grammar_inst.cpp>
#include <stan/lang/grammars/indexes_grammar_inst.cpp>
#include <stan/lang/grammars/whitespace_grammar_inst.cpp>

#include <stan/lang/grammars/semantic_actions_def.cpp>

#include <stan/lang/generator/expression_visgen.hpp>
#include <stan/lang/generator/generate_array_builder_adds.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_idxs.hpp>
#include <stan/lang/generator/generate_idxs_user.hpp>
#include <stan/lang/generator/generate_idx.hpp>
#include <stan/lang/generator/generate_idx_user.hpp>
#include <stan/lang/generator/idx_visgen.hpp>
#include <stan/lang/generator/idx_user_visgen.hpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
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

std::vector<stan::lang::block_var_decl>
parse_var_decls(std::string& input,
                  bool& pass,
                  std::ostream& err_msgs) {
  using boost::spirit::qi::expectation_failure;
  using boost::spirit::qi::phrase_parse;

  //  std::cout << "parsing: " << std::endl << input << std::endl;
  std::vector<std::string> search_path;
  search_path.push_back("foo");  
  std::stringstream ss(input);
  stan::io::program_reader reader(ss, "foo", search_path);

  typedef std::string::const_iterator input_iterator;
  typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;

  lp_iterator fwd_begin = lp_iterator(input.begin());
  lp_iterator fwd_end = lp_iterator(input.end());

  // block_var_decls_grammar args:  vm, msgs
  stan::lang::variable_map vm;
  std::stringstream msgs;

  // block_var_decls_grammar synthesis:  block_var_type
  std::vector<stan::lang::block_var_decl> parse_result;

  stan::lang::block_var_decls_grammar<lp_iterator> block_var_decls_grammar(vm, msgs, reader);
  stan::lang::whitespace_grammar<lp_iterator> whitesp_grammar(block_var_decls_grammar.error_msgs_);
  try {
    pass = phrase_parse(fwd_begin, fwd_end, block_var_decls_grammar,
                        whitesp_grammar, parse_result);
  } catch (const boost::spirit::qi::expectation_failure<lp_iterator>& e) {
    err_msgs << "expectation fail: " << e.what_ << std::endl;
  }
  err_msgs << block_var_decls_grammar.error_msgs_.str();
  if (fwd_begin != fwd_end) {
    std::basic_stringstream<char> unparsed_non_ws;
    unparsed_non_ws << boost::make_iterator_range(fwd_begin, fwd_end);
    err_msgs << "PARSER FAILED TO PARSE INPUT COMPLETELY"
             << std::endl
             << "STOPPED AT line "
             << get_line(fwd_begin)
             << ": "
             << std::endl
             << unparsed_non_ws.str()
             << std::endl;
  }
  return parse_result;
}

// block_array_type;
// corr_matrix_block_type;
// cov_matrix_block_type;
// matrix_block_type;
// ordered_block_type;
// positive_ordered_block_type;
// simplex_block_type;
// unit_vector_block_type;


TEST(Parser, parse_cholesky_corr_block_type) {
  std::string input("  int K;\n  cholesky_factor_corr[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("cholesky_factor_corr", ss.str());
}

TEST(Parser, parse_array_of_cholesky_corr_block_type) {
  std::string input("  int K;\n  cholesky_factor_corr[K] x[10, 10];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("2-dim array of cholesky_factor_corr", ss.str());
}

TEST(Parser, parse_cholesky_factor_block_type) {
  std::string input("  int K;\n  cholesky_factor_cov[K] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int", ss.str());

  ss.str(std::string());
  ss.clear();
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("cholesky_factor_cov", ss.str());
}
