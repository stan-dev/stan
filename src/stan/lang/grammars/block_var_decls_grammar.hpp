#ifndef STAN_LANG_GRAMMARS_BLOCK_VAR_DECLS_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_BLOCK_VAR_DECLS_GRAMMAR_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <stan/lang/grammars/expression_grammar.hpp>
#include <stan/lang/grammars/expression07_grammar.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/spirit/include/qi.hpp>
#include <string>
#include <sstream>
#include <vector>

namespace stan {
  namespace lang {

    template <typename Iterator>
    struct block_var_decls_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   boost::spirit::qi::locals<scope>,
                                   std::vector<block_var_decl>,
                                   whitespace_grammar<Iterator> > {
      block_var_decls_grammar(variable_map& var_map,
                        std::stringstream& error_msgs);

      variable_map& var_map_;
      std::stringstream& error_msgs_;
      expression_grammar<Iterator> expression_g;
      expression07_grammar<Iterator> expression07_g;  // disallows comparisons

      boost::spirit::qi::rule<Iterator,
                              expression(scope),
                              whitespace_grammar<Iterator> >
      def_r;

      boost::spirit::qi::rule<Iterator,
                              double_block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      double_decl_r;

      boost::spirit::qi::rule<Iterator,
                              double_block_type(scope),
                              whitespace_grammar<Iterator> >
      double_type_r;

      boost::spirit::qi::rule<Iterator,
                              std::string(),
                              whitespace_grammar<Iterator> >
      identifier_r;

      boost::spirit::qi::rule<Iterator,
                              std::string(),
                              whitespace_grammar<Iterator> >
      identifier_name_r;

      boost::spirit::qi::rule<Iterator,
                              int_block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      int_decl_r;

      boost::spirit::qi::rule<Iterator,
                              int_block_type(scope),
                              whitespace_grammar<Iterator> >
      int_type_r;

      boost::spirit::qi::rule<Iterator,
                              expression(scope),
                              whitespace_grammar<Iterator> >
      opt_def_r;

      boost::spirit::qi::rule<Iterator,
                              range(scope),
                              whitespace_grammar<Iterator> >
      range_brackets_double_r;

      boost::spirit::qi::rule<Iterator,
                              range(scope),
                              whitespace_grammar<Iterator> >
      range_brackets_int_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              block_var_decl(scope),
                              whitespace_grammar<Iterator> >
      var_decl_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<scope>,
                              std::vector<block_var_decl>,
                              whitespace_grammar<Iterator> >
      var_decls_r;
    };

  }
}
#endif
