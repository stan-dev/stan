#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <iomanip>
#include <iostream>
#include <istream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "stan/gm/ast.hpp"
#include "stan/gm/parser.hpp"
#include "stan/gm/generator.hpp"

int main() {
  static int SUCCESS_RC = 0;
  static int EXCEPTION_RC = -1;
  static int PARSE_FAIL_RC = -2;

  stan::gm::program prog;
  try {
    bool succeeded = stan::gm::parse(std::cin, "STDIN", prog); 
    if (!succeeded) {
      std::cout << "PARSE FAIL." << std::endl;
      return PARSE_FAIL_RC;
    }
  } catch(const std::exception& e) {
    std::cerr << "  EXCEPTION. " << e.what() << std::endl;
    return EXCEPTION_RC;
  }
  
  std::string model_name = "test_model";
  stan::gm::generate_cpp(prog,model_name,std::cout);

  return SUCCESS_RC;
}

