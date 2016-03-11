#ifndef STAN_LANG_GRAMMARS_EXPRESSION_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_EXPRESSION_GRAMMAR_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/expression07_grammar.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <boost/spirit/include/qi.hpp>
#include <sstream>

namespace stan {

  namespace lang {

    template <typename Iterator>
    struct expression07_grammar;

    template <typename Iterator>
    struct expression_grammar
      : public boost::spirit::qi::grammar<Iterator,
                                          expression(var_origin),
                                          whitespace_grammar<Iterator> > {
      variable_map& var_map_;
      std::stringstream& error_msgs_;
      expression07_grammar<Iterator> expression07_g;

      expression_grammar(variable_map& var_map,
                         std::stringstream& error_msgs);

      boost::spirit::qi::rule<Iterator,
                              expression(var_origin),
                              whitespace_grammar<Iterator> >
      expression_r;

      boost::spirit::qi::rule<Iterator,
                              expression(var_origin),
                              whitespace_grammar<Iterator> >
      expression09_r;

      boost::spirit::qi::rule<Iterator,
                              expression(var_origin),
                              whitespace_grammar<Iterator> >
      expression10_r;

      boost::spirit::qi::rule<Iterator,
                              expression(var_origin),
                              whitespace_grammar<Iterator> >
      expression14_r;
    };

  }
}
#endif
