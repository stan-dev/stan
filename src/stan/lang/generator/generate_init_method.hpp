#ifndef STAN_LANG_GENERATOR_GENERATE_INIT_METHOD_HPP
#define STAN_LANG_GENERATOR_GENERATE_INIT_METHOD_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_initializer.hpp>
#include <stan/lang/generator/init_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the <code>transform_inits</code> method for the
     * specified variable declarations to the specified stream.
     *
     * @param[in] vs variable declarations
     * @param[in,out] o stream for generating
     */
    void generate_init_method(const std::vector<var_decl>& vs,
                              std::ostream& o) {
      o << EOL;
      o << INDENT
        << "void transform_inits(const stan::io::var_context& context__,"
        << EOL;
      o << INDENT << "                     std::vector<int>& params_i__,"
        << EOL;
      o << INDENT << "                     std::vector<double>& params_r__,"
        << EOL;
      o << INDENT << "                     std::ostream* pstream__) const {"
        << EOL;
      o << INDENT2 << "stan::io::writer<double> "
        << "writer__(params_r__,params_i__);"
        << EOL;
      o << INDENT2 << "size_t pos__;" << EOL;
      o << INDENT2 << "(void) pos__; // dummy call to supress warning" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;
      o << INDENT2 << "std::vector<int> vals_i__;"
        << EOL;
      init_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);

      o << EOL
        << INDENT2 << "params_r__ = writer__.data_r();" << EOL;
      o << INDENT2 << "params_i__ = writer__.data_i();" << EOL;
      o << INDENT << "}" << EOL2;

      o << INDENT
        << "void transform_inits(const stan::io::var_context& context," << EOL;
      o << INDENT
        << "                     "
        << "Eigen::Matrix<double,Eigen::Dynamic,1>& params_r," << EOL;
      o << INDENT
        << "                     std::ostream* pstream__) const {" << EOL;
      o << INDENT << "  std::vector<double> params_r_vec;" << EOL;
      o << INDENT << "  std::vector<int> params_i_vec;" << EOL;
      o << INDENT
        << "  transform_inits(context, params_i_vec, params_r_vec, pstream__);"
        << EOL;
      o << INDENT << "  params_r.resize(params_r_vec.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    params_r(i) = params_r_vec[i];" << EOL;
      o << INDENT << "}" << EOL2;
    }

  }
}
#endif
