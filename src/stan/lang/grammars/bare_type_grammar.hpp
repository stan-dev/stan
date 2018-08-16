#ifndef STAN_LANG_GRAMMARS_BARE_TYPE_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_BARE_TYPE_GRAMMAR_HPP

#include <boost/spirit/include/qi.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>

#include <string>
#include <sstream>
#include <vector>

namespace stan {

  namespace lang {

    template <typename Iterator>
    struct bare_type_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   expr_type(),
                                   whitespace_grammar<Iterator> > {
      variable_map& var_map_;  // global info for function defs
      std::stringstream& error_msgs_;

      bare_type_grammar(variable_map& var_map,
                        std::stringstream& error_msgs);

      boost::spirit::qi::rule<Iterator,
                              expr_type(),
                              whitespace_grammar<Iterator> >
      bare_type_r;

      boost::spirit::qi::rule<Iterator,
                              base_expr_type(),
                              whitespace_grammar<Iterator> >
      type_identifier_r;

      boost::spirit::qi::rule<Iterator,
                              size_t(),
                              whitespace_grammar<Iterator> >
      array_dims_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::unused_type,
                              whitespace_grammar<Iterator> >
      end_bare_types_r;
    };

  }
}
#endif
