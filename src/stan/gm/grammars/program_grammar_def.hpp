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

#include <boost/format.hpp>
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

        using boost::phoenix::construct;
        using boost::phoenix::val;
        using boost::spirit::get_line;
        using boost::format;
        using std::setw;

        error_msgs << msg << std::endl;

        size_t idx_errline = 0;
        idx_errline = get_line(_where);

        if (idx_errline > 0) {
          error_msgs << "ERROR at line " << idx_errline << std::endl;

          std::basic_stringstream<char> sprogram;
          sprogram << boost::make_iterator_range (_begin, _end);

          // show error in context 2 lines before, 1 lines after
          size_t idx_errcol = 0;
          idx_errcol = get_column(_begin,_where) - 1;

          std::string lineno = "";
          format fmt_lineno("% 3d:    ");

          std::string line_2before = "";
          std::string line_before = "";
          std::string line_err = "";
          std::string line_after = "";            

          size_t idx_line = 0;
          size_t idx_before = idx_errline - 1;
          if (idx_before > 0) {
              // read lines up to error line, save 2 most recently read
              while (idx_before > idx_line) {
                line_2before = line_before;
                std::getline(sprogram,line_before);  
                idx_line++;
              }
              if (line_2before.length() > 0) {
                lineno = str(fmt_lineno % (idx_before - 1) );
                error_msgs << lineno << line_2before << std::endl;
              }
              lineno = str(fmt_lineno % idx_before);
              error_msgs << lineno << line_before << std::endl;
          }

          std::getline(sprogram,line_err);
          lineno = str(fmt_lineno % idx_errline);
          error_msgs << lineno << line_err << std::endl
                     << setw(idx_errcol + lineno.length()) << "^" << std::endl;
            
          if (!sprogram.eof()) {
            std::getline(sprogram,line_after);
            lineno = str(fmt_lineno % (idx_errline+1));
            error_msgs << lineno << line_after << std::endl;
          }
        }
        error_msgs << std::endl;
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
