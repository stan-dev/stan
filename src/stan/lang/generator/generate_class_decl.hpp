#ifndef STAN_LANG_GENERATOR_GENERATE_CLASS_DECL_HPP
#define STAN_LANG_GENERATOR_GENERATE_CLASS_DECL_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <ostream>
#include <string>

namespace stan {
namespace lang {

/**
 * Generate the specified name for the model class to the
 * specified stream.
 *
 * @param[in] model_name name of class
 * @param[in,out] o stream for generating
 */
void generate_class_decl(const std::string& model_name, std::ostream& o) {
  o << "class " << model_name << EOL
    << "  : public stan::model::model_base_crtp<" << model_name << "> {" << EOL;
}

}  // namespace lang
}  // namespace stan
#endif
