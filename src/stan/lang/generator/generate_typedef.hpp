#ifndef STAN_LANG_GENERATOR_GENERATE_TYPEDEF_HPP
#define STAN_LANG_GENERATOR_GENERATE_TYPEDEF_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <ostream>
#include <string>

namespace stan {
namespace lang {

/**
 * Generate a typedef statement for the specified type.
 *
 * @param[in] type type for definition
 * @param[in] abbrev abbreviation defined for type
 * @param[in,out] o stream for writing
 */
void generate_typedef(const std::string& type, const std::string& abbrev,
                      std::ostream& o) {
  o << "typedef"
    << " " << type << " " << abbrev << ";" << EOL2;
}
}  // namespace lang
}  // namespace stan
#endif
