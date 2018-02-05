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
    err_msgs << "PARSER EXPECTED: whitespace to end of file."
             << std::endl
             << "FOUND AT line "
             << get_line(fwd_begin)
             << ": "
             << std::endl
             << unparsed_non_ws.str()
             << std::endl;
  }
  return parse_result;
}


TEST(Parser, parse_empty) {
  std::string input("");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(0 == bvds.size());
}

TEST(Parser, parse_1) {
  std::string input("int<lower=0> x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  EXPECT_EQ("int< lower>", ss.str());
}

TEST(Parser, parse_2) {
  std::string input("  real<lower=0> x;\n  real<lower=2.1,upper=2.9> y;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());

  std::stringstream ss;
  stan::lang::write_block_var_type(ss, bvds[0].type());
  ss << std::endl;
  stan::lang::write_block_var_type(ss, bvds[1].type());
  EXPECT_EQ("real< lower>\nreal< lower, upper>", ss.str());
}

TEST(Parser, parse_3) {
  std::string input("  int x;\n  real y;\n  real<lower=2.1, upper=2.9> z;\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(3 == bvds.size());
}

TEST(Parser, parse_matrix) {
  std::string input("matrix<lower=0>[1,2] my_matrix;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_matrix2) {
  std::string input("matrix<lower=0.0, upper=1.0>[1,2] x;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_vector) {
  std::string input("vector[5] a;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_vector2) {
  std::string input("vector<lower=0.001>[5] b[10,20,30];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_vector3) {
  std::string input("vector<upper=0.001>[5] c[10,20,30];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_row_vector) {
  std::string input("row_vector[5] a;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_simplex) {
  std::string input("simplex[5] a;");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_array_1) {
  std::string input("int N[5];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_array_2) {
  std::string input("int<lower=1> N[5,5];");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(1 == bvds.size());
}

TEST(Parser, parse_1d_array_matrix) {
  std::string input("int x;\nmatrix[2,2] array_of_mat[100];\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
}

TEST(Parser, parse_2d_array_matrix) {
  std::string input("matrix[2,2] d1_array_of_mat[100];\n matrix[2,2] d2_array_of_mat[100,100];\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(input, pass, msgs);
  EXPECT_TRUE(pass);
  EXPECT_EQ(msgs.str(), std::string());
  EXPECT_TRUE(2 == bvds.size());
}
