#ifndef STAN_LANG_GRAMMARS_BLOCK_VAR_DECLS_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_BLOCK_VAR_DECLS_GRAMMAR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/block_var_decls_grammar.hpp>
#include <stan/lang/grammars/common_adaptors_def.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/phoenix/phoenix.hpp>
#include <boost/version.hpp>
#include <set>
#include <string>
#include <vector>

BOOST_FUSION_ADAPT_STRUCT(stan::lang::int_block_type,
                          (stan::lang::range, bounds_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::double_block_type,
                          (stan::lang::range, bounds_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::vector_block_type,
                          (stan::lang::range, bounds_)
                          (stan::lang::expression, N_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::row_vector_block_type,
                          (stan::lang::range, bounds_)
                          (stan::lang::expression, N_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::matrix_block_type,
                          (stan::lang::range, bounds_)
                          (stan::lang::expression, M_)
                          (stan::lang::expression, N_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::unit_vector_block_type,
                          (stan::lang::expression, K_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::simplex_block_type,
                          (stan::lang::expression, K_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::ordered_block_type,
                          (stan::lang::expression, K_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::positive_ordered_block_type,
                          (stan::lang::expression, K_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::cholesky_factor_block_type,
                          (stan::lang::expression, M_)
                          (stan::lang::expression, N_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::cholesky_corr_block_type,
                          (stan::lang::expression, K_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::cov_matrix_block_type,
                          (stan::lang::expression, K_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::corr_matrix_block_type,
                          (stan::lang::expression, K_) )

namespace stan {

  namespace lang {

    template <typename Iterator>
    block_var_decls_grammar<Iterator>::block_var_decls_grammar(
                                             variable_map& var_map,
                                             std::stringstream& error_msgs)
      : block_var_decls_grammar::base_type(block_var_decls_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression_g(var_map, error_msgs),
        expression07_g(var_map, error_msgs, expression_g) {
      using boost::spirit::qi::_1;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::eps;
      using boost::spirit::qi::lexeme;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::no_skip;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;
      using boost::spirit::qi::raw;

      using boost::spirit::qi::labels::_a;
      using boost::spirit::qi::labels::_b;
      using boost::spirit::qi::labels::_r1;

      using boost::phoenix::begin;
      using boost::phoenix::end;

      // _r1 var scope
      block_var_decls_r.name("block variable declarations");
      block_var_decls_r
        %= eps[set_var_scope_local_f(_b, model_name_origin)]
        > *(block_var_decl_r(_b));

      // for now - get this working then plug in array syntax
      // _r1 var scope
      block_var_decl_r.name("block variable declaration");
      block_var_decl_r(_r1)
        %= block_el_var_decl_r(_r1);
      
      // _r1 var scope
      // block_var_decl_r.name("block variable declaration");
      // block_var_decl_r
      //   %=  block_array_var_decl_r(_r1)
      //   | block_el_var_decl_r(_r1);
      // TODO:mitzi - eps to validate?

      // _r1 var scope
      // block_array_var_decl_r.name("block array variable declaration");
      // block_array_var_decl_r
      //   = raw[[((block_el_type_r(_r1)
      //            > array_dims_r(_r1))
      //           | (block_el_type_r(_r1)
      //              > identifier_r
      //              > array_dims_r(_r1)))]
      //         [set_block_array_type_f(_val, ... local vars ...
      //         ]
      //   [add_line_number_f(_val, begin(_1), end(_1))];
      // TODO:mitzi need to construct proper block_array_type, block_array_var_decl
      // > eps
      //          [validate_decl_constraints_f(_r1, _a, _val, _pass,
      //                                       boost::phoenix::ref(error_msgs_)),
      //           validate_definition_f(_r1, _val, _pass,
      //                                 boost::phoenix::ref(error_msgs_))]
      // > lit(';');

      // _a = error state local,
      // _r1 var scope
      // block_el_var_decl_r.name("block non-array variable declaration");
      // block_el_var_decl_r
      //   = raw[block_el_var_decl_sub_r(_r1)
      //         [add_block_var_f(_val, _1, boost::phoenix::ref(var_map_), _a, _r1,
      //                     boost::phoenix::ref(error_msgs))]]
      //   [add_line_number_f(_val, begin(_1), end(_1))];

      // block_el_var_decl_sub_r.name("block non-array variable declaration");
      // block_el_var_decl_sub_r(_r1)
      //   = block_el_type_r(_r1)
      //   > identifier_r
      //   > opt_def_r(_r1);

      // TODO:mitzi - figure out line numbers for var_decls and statements
      block_el_var_decl_r.name("block non-array variable declaration");
      block_el_var_decl_r
        = block_el_var_decl_sub_r(_r1)
          [add_block_var_f(_val, _1, boost::phoenix::ref(var_map_), _a, _r1,
                           boost::phoenix::ref(error_msgs))];

      block_el_var_decl_sub_r.name("block non-array variable declaration");
      block_el_var_decl_sub_r(_r1)
        = block_el_type_r(_r1)
        > identifier_r
        > opt_def_r(_r1);

      
      // _a = error state local,
      // _r1 var scope
      block_el_type_r.name("block non-array variable type");
      block_el_type_r
        = int_type_r(_r1)
          | double_type_r(_r1)
          | vector_type_r(_r1)
          | row_vector_type_r(_r1)
          | matrix_type_r(_r1)
          | unit_vector_type_r(_r1)
          | simplex_type_r(_r1)
          | ordered_type_r(_r1)
          | positive_ordered_type_r(_r1)
          | cholesky_factor_type_r(_r1)
          | cholesky_corr_type_r(_r1)
          | cov_matrix_type_r(_r1)
          | corr_matrix_type_r(_r1);

      // _r1 var scope
      int_type_r.name("integer declaration");
      int_type_r
        %= (lit("int")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > -range_brackets_int_r(_r1);

      double_type_r.name("real declaration");
      double_type_r
        %= (lit("real")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > -range_brackets_double_r(_r1);

      vector_type_r.name("vector declaration");
      vector_type_r
        %= (lit("vector")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > -range_brackets_double_r(_r1)
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      row_vector_type_r.name("row vector declaration");
      row_vector_type_r
        %= (lit("row_vector")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > -range_brackets_double_r(_r1)
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      matrix_type_r.name("matrix declaration");
      matrix_type_r
        %= (lit("matrix")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > -range_brackets_double_r(_r1)
        > lit('[')
        > int_data_expr_r(_r1) > lit(',') > int_data_expr_r(_r1)
        > lit(']');

      unit_vector_type_r.name("unit_vector declaration");
      unit_vector_type_r
        %= (lit("unit_vector")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      simplex_type_r.name("simplex declaration");
      simplex_type_r
        %= (lit("simplex")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      ordered_type_r.name("ordered declaration");
      ordered_type_r
        %= (lit("ordered")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      positive_ordered_type_r.name("positive_ordered declaration");
      positive_ordered_type_r
        %= (lit("positive_ordered")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      cholesky_factor_type_r.name("cholesky factor for symmetric,"
                                  " positive-def declaration");
      cholesky_factor_type_r
        %= (lit("cholesky_factor_cov")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('[')
        > int_data_expr_r(_r1) > -(lit(',') > int_data_expr_r(_r1))
        > lit(']');
        //        > eps
        //        [copy_square_cholesky_dimension_if_necessary_f(_val)];

      cholesky_corr_type_r.name("cholesky factor for"
                                " correlation matrix declaration");
      cholesky_corr_type_r
        %= (lit("cholesky_factor_corr")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      cov_matrix_type_r.name("covariance matrix declaration");
      cov_matrix_type_r
        %= (lit("cov_matrix")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      corr_matrix_type_r.name("correlation matrix declaration");
      corr_matrix_type_r
        %= (lit("corr_matrix")
            >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('[') > int_data_expr_r(_r1) > lit(']');

      // _r1 var scope
      int_data_expr_r.name("integer data expression");
      int_data_expr_r
        %= expression_g(_r1)
           [validate_int_data_only_expr_f(_1, _pass,
                                          boost::phoenix::ref(var_map_),
                                          boost::phoenix::ref(error_msgs_))];

      // _r1 var scope
      array_dims_r.name("array dimensions");
      array_dims_r %= lit('[') > (int_data_expr_r(_r1) % ',') > lit(']');

      // _r1 var scope
      opt_def_r.name("variable definition (optional)");
      opt_def_r %= -(lit('=') > expression_g(_r1));

      // _r1 var scope
      range_brackets_int_r.name("integer range expression pair, brackets");
      range_brackets_int_r
        = lit('<') [empty_range_f(_val, boost::phoenix::ref(error_msgs_))]
        >> (
            ((lit("lower")
              >> lit('=')
              >> expression07_g(_r1)
                 [set_int_range_lower_f(_val, _1, _pass,
                                        boost::phoenix::ref(error_msgs_))])
             >> -(lit(',')
                  >> lit("upper")
                  >> lit('=')
                  >> expression07_g(_r1)
                     [set_int_range_upper_f(_val, _1, _pass,
                                            boost::phoenix::ref(error_msgs_))]))
           |
           (lit("upper")
            >> lit('=')
            >> expression07_g(_r1)
               [set_int_range_upper_f(_val, _1, _pass,
                                      boost::phoenix::ref(error_msgs_))])
            )
        >> lit('>');

      // _r1 var scope
      range_brackets_double_r.name("real range expression pair, brackets");
      range_brackets_double_r
        = lit('<')[empty_range_f(_val, boost::phoenix::ref(error_msgs_))]
        > (
           ((lit("lower")
             > lit('=')
             > expression07_g(_r1)
               [set_double_range_lower_f(_val, _1, _pass,
                                         boost::phoenix::ref(error_msgs_))])
             > -(lit(',')
                 > lit("upper")
                 > lit('=')
                 > expression07_g(_r1)
                   [set_double_range_upper_f(_val, _1, _pass,
                                         boost::phoenix::ref(error_msgs_))]))
           |
           (lit("upper")
            > lit('=')
            > expression07_g(_r1)
              [set_double_range_upper_f(_val, _1, _pass,
                                        boost::phoenix::ref(error_msgs_))])
            )
        > lit('>');

      identifier_r.name("identifier");
      identifier_r
        %= identifier_name_r
           [validate_identifier_f(_val, _pass,
                                  boost::phoenix::ref(error_msgs_))];

      identifier_name_r.name("identifier subrule");
      identifier_name_r
        %= lexeme[char_("a-zA-Z")
                  >> *char_("a-zA-Z0-9_.")];
    }
  }


}
#endif
