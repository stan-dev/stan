#ifndef STAN__GM__PARSER__PROGRAM_GRAMMAR_DEF__HPP
#define STAN__GM__PARSER__PROGRAM_GRAMMAR_DEF__HPP

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <istream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

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
#include <boost/spirit/home/support/iterators/line_pos_iterator.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/program_grammar.hpp>
#include <stan/gm/grammars/functions_grammar.hpp>

namespace {
  // hack to pass pair into macro below to adapt; in namespace to hide
  struct DUMMY_STRUCT {
    typedef std::pair<std::vector<stan::gm::var_decl>,
                      std::vector<stan::gm::statement> > type;
  };
}


BOOST_FUSION_ADAPT_STRUCT(stan::gm::program,
                          (std::vector<stan::gm::function_decl_def>, function_decl_defs_)
                          (std::vector<stan::gm::var_decl>, data_decl_)
                          (DUMMY_STRUCT::type, derived_data_decl_)
                          (std::vector<stan::gm::var_decl>, parameter_decl_)
                          (DUMMY_STRUCT::type, derived_decl_)
                          (stan::gm::statement, statement_)
                          (DUMMY_STRUCT::type, generated_decl_) )


namespace stan {

  namespace gm {

    struct add_lp_var {
      template <typename T>
      struct result { typedef void type; };
      void operator()(variable_map& vm) const {
        vm.add("lp__",
               base_var_decl("lp__",std::vector<expression>(),DOUBLE_T),
               local_origin); // lp acts as a local where defined
      }
    };
    boost::phoenix::function<add_lp_var> add_lp_var_f;

    struct remove_lp_var {
      template <typename T>
      struct result { typedef void type; };
      void operator()(variable_map& vm) const {
        vm.remove("lp__");
      }
    };
    boost::phoenix::function<remove_lp_var> remove_lp_var_f;

    struct program_error {
      template <typename T1, typename T2, typename ,
        typename T4, typename T5, typename T6, typename T7>
      struct result { typedef void type; };

      template <class Iterator, class I>
      void operator()(
        Iterator _begin, 
        Iterator _end, 
        Iterator _where, 
        I const& _info,
        std::string msg,
        variable_map& vm,
        std::stringstream& error_msgs) const {

        error_msgs << msg
                   << std::endl;

        using boost::phoenix::construct;
        using boost::phoenix::val;

        std::basic_stringstream<char> pre_error_section;
        pre_error_section << boost::make_iterator_range (_begin, _where);
        char last_char;
        std::string correct_section = "";
        while (!pre_error_section.eof()) {
          last_char = (char)pre_error_section.get();
          correct_section += last_char;
        }

        size_t indx = correct_section.size();
        correct_section = correct_section.erase(indx-1, indx);

        //
        //  Clean up whatever is before the error occurred
        //
        //  Would be better to use the parser to select which 
        //  section in the stan file contains the parsing error.
        //

        std::vector<std::string> sections;
        sections.push_back("generated");
        sections.push_back("model");
        sections.push_back("transformed");
        sections.push_back("parameter");
        sections.push_back("data");

        //bool found_section = false; // FIXME: do something with found_section
        indx = 0;

        for (size_t i = 0; i < sections.size(); ++i) {
          std::string section = sections[i];
          indx = correct_section.find(section);
          if (!(indx == std::string::npos)) {
            if (i == 2) {
              // Check which transformed block we're dealing with.
              // If there is another transformed section, it must be
              // a 'transformed parameters' section
              size_t indx2 = correct_section.find("transformed", indx + 5);
              if (!(indx2 == std::string::npos)) {
                indx = indx2;
              } else {
                // No second transformed section, but maybe there
                // is a parameter block?
                indx2 = correct_section.find("parameters", indx2);
                if (!(indx2 == std::string::npos)) {
                  indx = indx2;
                }
                // Ok, we found a 'transformed data' block.
                // indx is pointing at it.
              }
            }
            //found_section = true;
            correct_section = correct_section.erase(0, indx);
            break;
          }
        }

        //
        //  Clean up whatever comes after the error occurred
        //
        std::basic_stringstream<char> error_section;
        error_section << boost::make_iterator_range (_where, _end);
        last_char = ' ';
        std::string rest_of_section = "";
        while (!error_section.eof() && !(last_char == '}')) {
          last_char = (char)error_section.get();
          rest_of_section += last_char;
          //std::cout << rest_of_section.size() << std::endl;
          if (error_section.eof() && rest_of_section.size() == 1) {
            rest_of_section = "'end of file'";
          }
        }

        if (!(get_line(_where) == std::string::npos)) {
          error_msgs
            << std::endl
            << "LOCATION OF PARSING ERROR (line = "
            << get_line(_where)
            << ", position = "
            << get_column(_begin, _where) - 1
            <<  "):"
            << std::endl
            << std::endl
            << "PARSED:"
            << std::endl
            << std::endl
            << correct_section
            << std::endl
            << std::endl
            << "EXPECTED: " << _info
            << " BUT FOUND: " 
            << std::endl
            << std::endl
            << rest_of_section
            << std::endl;
        }
      }
    };
    boost::phoenix::function<program_error> program_error_f;

    template <typename Iterator>
    program_grammar<Iterator>::program_grammar(const std::string& model_name) 
        : program_grammar::base_type(program_r),
          model_name_(model_name),
          var_map_(),
          error_msgs_(),
          expression_g(var_map_,error_msgs_),
          var_decls_g(var_map_,error_msgs_),
          statement_g(var_map_,error_msgs_),
          functions_g(var_map_,error_msgs_) {

        using boost::spirit::qi::eps;
        using boost::spirit::qi::lit;

        // add model_name to var_map with special origin and no 
        var_map_.add(model_name,
                     base_var_decl(),
                     model_name_origin);

        program_r.name("program");
        program_r 
          %= -functions_g
          > -data_var_decls_r
          > -derived_data_var_decls_r
          > -param_var_decls_r
          // scope lp__ to "transformed params" and "model" only
          > eps[add_lp_var_f(boost::phoenix::ref(var_map_))]
          > -derived_var_decls_r
          > model_r
          > eps[remove_lp_var_f(boost::phoenix::ref(var_map_))]
          > -generated_var_decls_r
          ;

        model_r.name("model declaration");
        model_r 
          %= lit("model")
          > statement_g(true,local_origin,false)  // assign only to locals
          ;

        data_var_decls_r.name("data variable declarations");
        data_var_decls_r
          %= lit("data")
          > lit('{')
          > var_decls_g(true,data_origin) // +constraints
          > lit('}');

        derived_data_var_decls_r.name("transformed data block");
        derived_data_var_decls_r
          %= ( lit("transformed")
               >> lit("data") )
          > lit('{')
          > var_decls_g(true,transformed_data_origin)  // -constraints
          > *statement_g(false,transformed_data_origin,false) // -sampling
          > lit('}');

        param_var_decls_r.name("parameter variable declarations");
        param_var_decls_r
          %= lit("parameters")
          > lit('{')
          > var_decls_g(true,parameter_origin) // +constraints
          > lit('}');

        derived_var_decls_r.name("derived variable declarations");
        derived_var_decls_r
          %= ( lit("transformed")
               >> lit("parameters") )
          > lit('{')
          > var_decls_g(true,transformed_parameter_origin) // -constraints
          > *statement_g(false,transformed_parameter_origin,false) // -sampling
          > lit('}');

        generated_var_decls_r.name("generated variable declarations");
        generated_var_decls_r
          %= lit("generated")
          > lit("quantities")
          > lit('{')
          > var_decls_g(true,derived_origin) // -constraints
          > *statement_g(false,derived_origin,false) // -sampling
          > lit('}');

        using boost::spirit::qi::on_error;
        using boost::spirit::qi::rethrow;
        using namespace boost::spirit::qi::labels;

        on_error<rethrow>(
          program_r,
          program_error_f(
            _1, _2, _3, _4 ,
            "",
            boost::phoenix::ref(var_map_),
            boost::phoenix::ref(error_msgs_)
          )
        ); 
    }

  }
}

#endif
