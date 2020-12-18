#ifdef USE_STANC3

#ifndef STAN_LANG_GENERATOR_GENERATE_MODEL_COMPILE_INFO_METHOD_HPP
#define STAN_LANG_GENERATOR_GENERATE_MODEL_COMPILE_INFO_METHOD_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <ostream>
#include <string>

namespace stan {
namespace lang {

/**
 * Generate the <code>model_compile_info</code> method on the specified stream.
 *
 * @param[in,out] o stream for generating
 */
void generate_model_compile_info_method(std::ostream& o) {
  o << INDENT << "std::vector<std::string> model_compile_info() const {" << EOL
    << INDENT2 << "std::vector<std::string> stanc_info;" << EOL << INDENT2
    << "stanc_info.push_back(\"stanc_version = stanc2\");" << EOL << INDENT2
    << "return stanc_info;" << EOL << INDENT << "}" << EOL2;
}

}  // namespace lang
}  // namespace stan
#endif

#endif
