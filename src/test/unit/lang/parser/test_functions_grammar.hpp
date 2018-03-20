#ifndef STAN_LANG_GRAMMARS_TEST_FUNCTIONS_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_TEST_FUNCTIONS_GRAMMAR_HPP

#include <stan/io/program_reader.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/functions_grammar.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <stan/lang/grammars/bare_type_grammar.hpp>
#include <stan/lang/grammars/statement_grammar.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/spirit/include/qi.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

namespace stan {

  namespace lang {

    template <typename Iterator>
    struct test_functions_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   std::vector<function_decl_def>(),
                                   whitespace_grammar<Iterator> > {
      const io::program_reader& reader_;
      variable_map var_map_;
      std::set<std::pair<std::string,
                         function_signature_t> > functions_declared_;
      std::set<std::pair<std::string,
                         function_signature_t> > functions_defined_;
      std::stringstream& error_msgs_;
      statement_grammar<Iterator> statement_g;
      bare_type_grammar<Iterator> bare_type_g;
      functions_grammar<Iterator> functions_g;

      test_functions_grammar(const io::program_reader& reader,
                                   variable_map& var_map,
                                   std::stringstream& error_msgs);

      boost::spirit::qi::rule<Iterator,
                              std::vector<function_decl_def>(),
                              whitespace_grammar<Iterator> >
      test_functions_r;
    };

  }
}
#endif
