#ifndef STAN_LANG_GENERATOR_WRITE_ARRAY_VARS_VISGEN_HPP
#define STAN_LANG_GENERATOR_WRITE_ARRAY_VARS_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    /**
     * Visitor for generating code to fill the <code>vars__</code>
     * accumulator with variable names.
     */
    struct write_array_vars_visgen : public visgen {
      /**
       * Construct a variable array writer visitor for the specified
       * stream.
       *
       * @param[in,out] o stream for generating
       */
      explicit write_array_vars_visgen(std::ostream& o) : visgen(o) { }

      /**
       * Write a variable with the specified name, array dimension
       * sizes, and matrix dimesnion sizes.
       *
       * @param name variable name
       * @param arraydims array dimension sizes
       * @param matdims matrix dimension sizes
       */
      void write_array(const std::string& name,
                       const std::vector<expression>& arraydims,
                       const std::vector<expression>& matdims) const {
        std::vector<expression> dims(arraydims);
        for (size_t i = 0; i < matdims.size(); ++i)
          dims.push_back(matdims[i]);
        if (dims.size() == 0) {
          o_ << INDENT2 << "vars__.push_back(" << name << ");" << EOL;
          return;
        }
        // for (size_t i = 0; i < dims.size(); ++i) {
        for (size_t i = dims.size(); i > 0; ) {
          --i;
          generate_indent((dims.size() - i) + 1, o_);
          o_ << "for (int k_" << i << "__ = 0;"
             << " k_" << i << "__ < ";
          generate_expression(dims[i], o_);
          o_ << "; ++k_" << i << "__) {" << EOL;
        }

        generate_indent(dims.size() + 2, o_);
        o_ << "vars__.push_back(" << name;
        if (arraydims.size() > 0) {
          o_ << '[';
          for (size_t i = 0; i < arraydims.size(); ++i) {
            if (i > 0) o_ << "][";
            o_ << "k_" << i << "__";
          }
          o_ << ']';
        }
        if (matdims.size() > 0) {
          o_ << "(k_" << arraydims.size() << "__";
          if (matdims.size() > 1)
            o_ << ", k_" << (arraydims.size() + 1) << "__";
          o_ << ")";
        }
        o_ << ");" << EOL;

        for (size_t i = dims.size(); i > 0; --i) {
          generate_indent(i + 1, o_);
          o_ << "}" << EOL;
        }
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        write_array(x.name_, x.dims_, EMPTY_EXP_VECTOR);
      }

      void operator()(const double_var_decl& x) const {
        write_array(x.name_, x.dims_, EMPTY_EXP_VECTOR);
      }

      void operator()(const vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }

      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.N_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }

      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.M_);
        matdims.push_back(x.N_);
        write_array(x.name_, x.dims_, matdims);
      }

      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }

      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }

      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.M_);
        matdims.push_back(x.N_);
        write_array(x.name_, x.dims_, matdims);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }
    };

  }
}
#endif
