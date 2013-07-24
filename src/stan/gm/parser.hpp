#ifndef __STAN__GM__PARSER__PARSER__HPP__
#define __STAN__GM__PARSER__PARSER__HPP__

#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
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

#include <stan/gm/ast.hpp>

#include <stan/gm/grammars/program_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>

namespace stan {

  namespace gm {

    bool is_space(char c) {
      return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    }

    bool is_nonempty(std::string& s) {
      for (size_t i = 0; i < s.size(); ++i)
        if (!is_space(s[i]))
          return true;
      return false;
    }

    // Cut and paste source for iterator & reporting pattern:
    // http://boost-spirit.com/home/articles/qi-example
    //                 /tracking-the-input-position-while-parsing/
    // http://boost-spirit.com/dl_more/parsing_tracking_position
    //                 /stream_iterator_errorposition_parsing.cpp
    inline bool parse(std::ostream* output_stream,
                      std::istream& input, 
                      const std::string& filename, 
                      const std::string& model_name,
                      program& result) {
      namespace classic = boost::spirit::classic;

      using boost::spirit::classic::position_iterator2;
      using boost::spirit::multi_pass;
      using boost::spirit::make_default_multi_pass;
      using std::istreambuf_iterator;

      using boost::spirit::qi::expectation_failure;
      using boost::spirit::classic::file_position_base;
      using boost::spirit::qi::phrase_parse;


      // iterate over stream input
      typedef istreambuf_iterator<char> base_iterator_type;
      typedef multi_pass<base_iterator_type>  forward_iterator_type;
      typedef position_iterator2<forward_iterator_type> pos_iterator_type;

      base_iterator_type in_begin(input);
      
      forward_iterator_type fwd_begin = make_default_multi_pass(in_begin);
      forward_iterator_type fwd_end;
      
      pos_iterator_type position_begin(fwd_begin, fwd_end, filename);
      pos_iterator_type position_end;
      
      program_grammar<pos_iterator_type> prog_grammar(model_name);
      whitespace_grammar<pos_iterator_type> whitesp_grammar;
      
      bool parse_succeeded = false;
      try {
        parse_succeeded = phrase_parse(position_begin, 
                                       position_end,
                                       prog_grammar,
                                       whitesp_grammar,
                                       result);
        std::string diagnostics = prog_grammar.error_msgs_.str();
        if (output_stream && is_nonempty(diagnostics)) {
          *output_stream << "DIAGNOSTIC(S) FROM PARSER:" 
                         << std::endl
                         << diagnostics 
                         << std::endl;
        }
      } catch (const expectation_failure<pos_iterator_type>& e) {
        const file_position_base<std::string>& pos = e.first.get_position();
        std::stringstream msg;
        msg << "EXPECTATION FAILURE LOCATION: file=" << pos.file
            << "; line=" << pos.line 
            << ", column=" << pos.column 
            << std::endl;
        msg << std::endl << e.first.get_currentline() 
            << std::endl;
        for (int i = 2; i < pos.column; ++i)
          msg << ' ';
        msg << " ^-- here" 
            << std::endl << std::endl;
        std::string diagnostics = prog_grammar.error_msgs_.str();
        if (output_stream && is_nonempty(diagnostics)) {
          msg << std::endl
              << "DIAGNOSTIC(S) FROM PARSER:"
              << diagnostics
              << std::endl;
        }
        throw std::invalid_argument(msg.str());

      } catch (const std::runtime_error& e) {
        std::stringstream msg;
        msg << "LOCATION: unknown" << std::endl;

        msg << "DIAGNOSTICS FROM PARSER:" << std::endl;
        msg << prog_grammar.error_msgs_.str() << std::endl << std::endl;
        throw std::invalid_argument(msg.str());
      }
      
      bool consumed_all_input = (position_begin == position_end); 
      bool success = parse_succeeded && consumed_all_input;

      if (!success) {      
        std::stringstream msg;
        if (!parse_succeeded)
          msg << "PARSE DID NOT SUCCEED." << std::endl; 
        if (!consumed_all_input)
          msg << "ERROR: non-whitespace beyond end of program:" << std::endl;
        
        const file_position_base<std::string>& pos 
          = position_begin.get_position();
        msg << "LOCATION: file=" << pos.file
            << "; line=" << pos.line
            << ", column=" << pos.column
            << std::endl;
        msg << position_begin.get_currentline() 
            << std::endl;
        for (int i = 2; i < pos.column; ++i)
          msg << ' ';
        msg << " ^-- starting here" 
            << std::endl << std::endl;

        msg << "DIAGNOSTICS FROM PARSER:" << std::endl;
        msg << prog_grammar.error_msgs_.str() << std::endl << std::endl;

        throw std::invalid_argument(msg.str());
      }
      return true;
    }


  }
}

#endif
