#ifndef STAN_LANG_GENERATOR_GENERATE_DIMS_METHOD_HPP
#define STAN_LANG_GENERATOR_GENERATE_DIMS_METHOD_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/write_dims_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate the <code>get_dims</code> method for the parameters,
     * transformed parameters, and generated quantities, using the
     * specified program and generating to the specified stream.
     *
     * @param[in] prog program from which to generate
     * @param[in,out] o stream for generating
     */
    void generate_dims_method(const program& prog, std::ostream& o) {
      write_dims_visgen vis(o);
      o << EOL << INDENT
        << "void get_dims(std::vector<std::vector<size_t> >& dimss__) const {"
        << EOL;
      o << INDENT2 << "dimss__.resize(0);" << EOL;
      o << INDENT2 << "std::vector<size_t> dims__;" << EOL;
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i)
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i)
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      o << INDENT << "}" << EOL2;
    }

  }
}
#endif
