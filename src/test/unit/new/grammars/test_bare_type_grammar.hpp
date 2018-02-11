#ifndef STAN_LANG_GRAMMARS_TEST_BARE_TYPE_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_TEST_BARE_TYPE_GRAMMAR_HPP

#include <stan/io/program_reader.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <stan/lang/grammars/bare_type_grammar.hpp>
#include <boost/spirit/include/qi.hpp>
#include <string>
#include <sstream>
#include <utility>

namespace stan {

  namespace lang {

    template <typename Iterator>
    struct test_bare_type_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   bare_expr_type,
                                   whitespace_grammar<Iterator> > {
      const io::program_reader& reader_;
      std::stringstream& error_msgs_;
      bare_type_grammar<Iterator> bare_type_g;

      test_bare_type_grammar(const io::program_reader& reader,
                                   std::stringstream& error_msgs);

      boost::spirit::qi::rule<Iterator,
                              bare_expr_type,
                              whitespace_grammar<Iterator> >
      test_bare_type_r;
    };

  }
}
#endif
