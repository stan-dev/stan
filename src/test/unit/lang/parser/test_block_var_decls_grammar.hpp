#ifndef STAN_LANG_GRAMMARS_TEST_BLOCK_VAR_DECLS_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_TEST_BLOCK_VAR_DECLS_GRAMMAR_HPP

#include <stan/io/program_reader.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <stan/lang/grammars/expression_grammar.hpp>
#include <stan/lang/grammars/block_var_decls_grammar.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/spirit/include/qi.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

namespace stan {

  namespace lang {

    template <typename Iterator>
    struct test_block_var_decls_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   boost::spirit::qi::locals<scope>,
                                   std::vector<block_var_decl>,
                                   whitespace_grammar<Iterator> > {
      const io::program_reader& reader_;
      variable_map var_map_;
      std::stringstream& error_msgs_;
      block_var_decls_grammar<Iterator> block_var_decls_g;

      test_block_var_decls_grammar(const io::program_reader& reader,
                                   variable_map& var_map,
                                   std::stringstream& error_msgs);

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<scope>,
                              std::vector<block_var_decl>,
                              whitespace_grammar<Iterator> >
      test_block_var_decls_r;
    };

  }
}
#endif
