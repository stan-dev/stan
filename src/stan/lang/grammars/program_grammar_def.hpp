#ifndef STAN_LANG_GRAMMARS_PROGRAM_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_PROGRAM_GRAMMAR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/program_grammar.hpp>
#include <stan/lang/grammars/semantic_actions.hpp>
#include <boost/format.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/support/iterators/line_pos_iterator.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/qi.hpp>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
  // hack to pass pair into macro below to adapt; in namespace to hide
  struct DUMMY_STRUCT {
    typedef std::pair<std::vector<stan::lang::var_decl>,
                      std::vector<stan::lang::statement> > type;
  };
}

BOOST_FUSION_ADAPT_STRUCT(stan::lang::program,
                          (std::vector<stan::lang::function_decl_def>,
                           function_decl_defs_)
                          (std::vector<stan::lang::var_decl>, data_decl_)
                          (DUMMY_STRUCT::type, derived_data_decl_)
                          (std::vector<stan::lang::var_decl>, parameter_decl_)
                          (DUMMY_STRUCT::type, derived_decl_)
                          (stan::lang::statement, statement_)
                          (DUMMY_STRUCT::type, generated_decl_) )


namespace stan {

  namespace lang {

    template <typename Iterator>
    program_grammar<Iterator>::program_grammar(const std::string& model_name,
                                               bool allow_undefined)
        : program_grammar::base_type(program_r),
          model_name_(model_name),
          var_map_(),
          error_msgs_(),
          expression_g(var_map_, error_msgs_),
          var_decls_g(var_map_, error_msgs_),
          statement_g(var_map_, error_msgs_),
          functions_g(var_map_, error_msgs_, allow_undefined) {
        using boost::spirit::qi::eps;
        using boost::spirit::qi::lit;
        using boost::spirit::qi::on_error;
        using boost::spirit::qi::rethrow;
        using boost::spirit::qi::_1;
        using boost::spirit::qi::_2;
        using boost::spirit::qi::_3;
        using boost::spirit::qi::_4;

        // add model_name to var_map with special origin
        var_map_.add(model_name, base_var_decl(), model_name_origin);

        program_r.name("program");
        program_r
          %= -functions_g
          > -data_var_decls_r
          > -derived_data_var_decls_r
          > -param_var_decls_r
          > eps[add_lp_var_f(boost::phoenix::ref(var_map_))]
          > -derived_var_decls_r
          > model_r
          > eps[remove_lp_var_f(boost::phoenix::ref(var_map_))]
          > -generated_var_decls_r;

        model_r.name("model declaration (or perhaps an earlier block)");
        model_r
          %= lit("model")
          > statement_g(true, local_origin, false, false);

        end_var_decls_r.name(
            "one of the following:\n"
            "  a variable declaration, beginning with type,\n"
            "      (int, real, vector, row_vector, matrix, unit_vector,\n"
            "       simplex, ordered, positive_ordered,\n"
            "       corr_matrix, cov_matrix,\n"
            "       cholesky_corr, cholesky_cov\n"
            "  or '}' to close variable declarations");
        end_var_decls_r %= lit('}');

        end_var_decls_statements_r.name(
           "one of the following:\n"
           "  a variable declaration, beginning with type\n"
           "      (int, real, vector, row_vector, matrix, unit_vector,\n"
           "       simplex, ordered, positive_ordered,\n"
           "       corr_matrix, cov_matrix,\n"
           "       cholesky_corr, cholesky_cov\n"
           "  or a <statement>\n"
           "  or '}' to close variable declarations and definitions");
        end_var_decls_statements_r %= lit('}');

        end_var_definitions_r.name("expected another statement or '}'"
                                   " to close declarations");
        end_var_definitions_r %= lit('}');

        data_var_decls_r.name("data variable declarations");
        data_var_decls_r
          %= (lit("data")
              > lit('{'))
          >  var_decls_g(true, data_origin)  // +constraints
          > end_var_decls_r;

        derived_data_var_decls_r.name("transformed data block");
        derived_data_var_decls_r
          %= ((lit("transformed")
               >> lit("data"))
              > lit('{'))
          > var_decls_g(true, transformed_data_origin)  // -constraints
          > ((statement_g(false, transformed_data_origin, false, false)
              > *statement_g(false, transformed_data_origin, false, false)
              > end_var_definitions_r)
             | (*statement_g(false, transformed_data_origin, false, false)
                > end_var_decls_statements_r));

        param_var_decls_r.name("parameter variable declarations");
        param_var_decls_r
          %= (lit("parameters")
              > lit('{'))
          > var_decls_g(true, parameter_origin)  // +constraints
          > end_var_decls_r;

        derived_var_decls_r.name("derived variable declarations");
        derived_var_decls_r
          %= (lit("transformed")
              > lit("parameters")
              > lit('{'))
          > var_decls_g(true, transformed_parameter_origin)
          > *statement_g(false, transformed_parameter_origin, false, false)
          > end_var_decls_statements_r;

        generated_var_decls_r.name("generated variable declarations");
        generated_var_decls_r
          %= (lit("generated")
              > lit("quantities")
              > lit('{'))
          > var_decls_g(true, derived_origin)
          > *statement_g(false, derived_origin, false, false)
          > end_var_decls_statements_r;

        on_error<rethrow>(program_r,
                          program_error_f(_1, _2, _3,
                                          boost::phoenix::ref(var_map_),
                                          boost::phoenix::ref(error_msgs_)));
    }

  }
}
#endif
