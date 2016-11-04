#ifndef STAN_LANG_GRAMMARS_VAR_DECLS_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_VAR_DECLS_GRAMMAR_HPP

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
    struct var_decls_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   boost::spirit::qi::locals<bool>,
                                   std::vector<var_decl>(bool, var_origin),
                                   whitespace_grammar<Iterator> > {
      var_decls_grammar(variable_map& var_map,
                        std::stringstream& error_msgs);

      variable_map& var_map_;
      std::stringstream& error_msgs_;
      expression_grammar<Iterator> expression_g;
      expression07_grammar<Iterator> expression07_g;  // disallows comparisons

      boost::spirit::qi::rule<Iterator,
                              cholesky_factor_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      cholesky_factor_decl_r;

      boost::spirit::qi::rule<Iterator,
                              cholesky_corr_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      cholesky_corr_decl_r;

      boost::spirit::qi::rule<Iterator,
                              corr_matrix_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      corr_matrix_decl_r;

      boost::spirit::qi::rule<Iterator,
                              cov_matrix_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      cov_matrix_decl_r;

      boost::spirit::qi::rule<Iterator,
                              expression(var_origin),
                              whitespace_grammar<Iterator> >
      def_r;

      boost::spirit::qi::rule<Iterator,
                              expression(var_origin),
                              whitespace_grammar<Iterator> >
      int_data_expr_r;

      boost::spirit::qi::rule<Iterator,
                              std::vector<expression>(var_origin),
                              whitespace_grammar<Iterator> >
      dims_r;

      boost::spirit::qi::rule<Iterator,
                              expression(var_origin),
                              whitespace_grammar<Iterator> >
      dim1_r;

      boost::spirit::qi::rule<Iterator,
                              double_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      double_decl_r;

      boost::spirit::qi::rule<Iterator,
                              std::string(),
                              whitespace_grammar<Iterator> >
      identifier_r;

      boost::spirit::qi::rule<Iterator,
                              std::string(),
                              whitespace_grammar<Iterator> >
      identifier_name_r;

      boost::spirit::qi::rule<Iterator,
                              int_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      int_decl_r;

      boost::spirit::qi::rule<Iterator,
                              matrix_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      matrix_decl_r;


      boost::spirit::qi::rule<Iterator,
                              expression(var_origin),
                              whitespace_grammar<Iterator> >
      opt_def_r;

      boost::spirit::qi::rule<Iterator,
                              std::vector<expression>(var_origin),
                              whitespace_grammar<Iterator> >
      opt_dims_r;

      boost::spirit::qi::rule<Iterator,
                              ordered_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      ordered_decl_r;

      boost::spirit::qi::rule<Iterator,
                              positive_ordered_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      positive_ordered_decl_r;

      boost::spirit::qi::rule<Iterator,
                              range(var_origin),
                              whitespace_grammar<Iterator> >
      range_brackets_double_r;

      boost::spirit::qi::rule<Iterator,
                              range(var_origin),
                              whitespace_grammar<Iterator> >
      range_brackets_int_r;

      boost::spirit::qi::rule<Iterator,
                              row_vector_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      row_vector_decl_r;

      boost::spirit::qi::rule<Iterator,
                              unit_vector_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      unit_vector_decl_r;

      boost::spirit::qi::rule<Iterator,
                              simplex_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      simplex_decl_r;

      boost::spirit::qi::rule<Iterator,
                              vector_var_decl(var_origin),
                              whitespace_grammar<Iterator> >
      vector_decl_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              var_decl(bool, var_origin),
                              whitespace_grammar<Iterator> >
      var_decl_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::locals<bool>,
                              std::vector<var_decl>(bool, var_origin),
                              whitespace_grammar<Iterator> >
      var_decls_r;
    };

  }
}
#endif
