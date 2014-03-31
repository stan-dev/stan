#ifndef STAN__GM__PARSER__BARE_TYPE_GRAMMAR_DEF_HPP__
#define STAN__GM__PARSER__BARE_TYPE_GRAMMAR_DEF_HPP__

#include <boost/spirit/include/qi.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <boost/spirit/include/version.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/bare_type_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>

BOOST_FUSION_ADAPT_STRUCT(stan::gm::expr_type,
                          (stan::gm::base_expr_type, base_type_)
                          (size_t, num_dims_) );

namespace stan {

  namespace gm {

     template <typename Iterator>
     bare_type_grammar<Iterator>::bare_type_grammar(variable_map& var_map,
                                                    std::stringstream& error_msgs)
       : bare_type_grammar::base_type(bare_type_r),
         var_map_(var_map),
         error_msgs_(error_msgs)
     {
      using boost::spirit::qi::_1;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::eps;
      using boost::spirit::qi::lexeme;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::no_skip;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;

      using boost::spirit::qi::labels::_a;
      using boost::spirit::qi::labels::_r1;
      using boost::spirit::qi::labels::_r2;

      using boost::spirit::qi::on_error;
      using boost::spirit::qi::fail;
      using boost::spirit::qi::rethrow;
      using namespace boost::spirit::qi::labels;

      bare_type_r.name("bare type");
      bare_type_r
        %= 
        type_identifier_r
        >> array_dims_r
        ;
      
      type_identifier_r
        %= 
        lit("void")[_val = VOID_T]
        | lit("int")[_val = INT_T]
        | lit("real")[_val = DOUBLE_T]
        | lit("vector")[_val = VECTOR_T]
        | lit("row_vector")[_val = ROW_VECTOR_T]
        | lit("matrix")[_val = MATRIX_T]
        ;
      
      array_dims_r
        %= 
        eps[_val = 0]
        >> - ( lit('[')[_val = 1]
               >> *(lit(',')[_val += 1])
               >> lit(']') )
        ;

     }
  }
}
#endif
