#ifndef STAN_LANG_GENERATOR_SET_PARAM_RANGES_VISGEN_HPP
#define STAN_LANG_GENERATOR_SET_PARAM_RANGES_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    /**
     * Visitor for generating ranges output for parameters used to set
     * local variables <code>param_ranges_i__</code> and
     * <code>param_ranges_r__</code> and <code>num_params_i__</code>
     * and <code>num_params_r__</code>.
     */
    struct set_param_ranges_visgen : public visgen {
      /**
       * Construct a visitor for generating ranges that writes to the
       * specified stream.
       *
       * @param[in,out] o stream for generating
       */
      explicit set_param_ranges_visgen(std::ostream& o)
        : visgen(o) {
      }

      /**
       * Generate code to increment the integer or real numbers of
       * parameters <code>num_params_i__</code> and
       * <code>num_params_r__</code>
       *
       * @param[in] dims sequence of dimensions
       */
      void generate_increment_i(const std::vector<expression>& dims) const {
        if (dims.size() == 0) {
          o_ << INDENT2 << "++num_params_i__;" << EOL;
          return;
        }
        o_ << INDENT2 << "num_params_r__ += ";
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0) o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }

      /**
       * Generate code to increment the number of real parameters
       * <code>num_params_r__</code>.
       *
       * @param[in] dims sequence of dimensions
       */
      void generate_increment(std::vector<expression> dims) const {
        if (dims.size() == 0) {
          o_ << INDENT2 << "++num_params_r__;" << EOL;
          return;
        }
        o_ << INDENT2 << "num_params_r__ += ";
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0) o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }


      /**
       * Generate code to increment the number of real parameters
       * <code>num_params_r__</code> for vector and row vector types.
       *
       * @param[in] K number of vector or row vector dimensions
       * @param[in] dims sequence of dimensions
       */
      void generate_increment(const expression& K,
                              const std::vector<expression>& dims) const {
        o_ << INDENT2 << "num_params_r__ += ";
        generate_expression(K, o_);
        for (size_t i = 0; i < dims.size(); ++i) {
          o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }

      /**
       * Generate code to increment the number of real parameters
       * <code>num_params_r__</code> for matrix types
       *
       * @param[in] M number of rows
       * @param[in] N number of columns
       * @param[in] dims sequence of dimensions
       */
      void generate_increment(const expression& M, const expression& N,
                              const std::vector<expression>& dims) const {
        o_ << INDENT2 << "num_params_r__ += ";
        generate_expression(M, o_);
        o_ << " * ";
        generate_expression(N, o_);
        for (size_t i = 0; i < dims.size(); ++i) {
          o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment_i(x.dims_);
        // for loop for ranges
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "for (size_t i_" << i << "__ = 0; ";
          o_ << "i_" << i << "__ < ";
          generate_expression(x.dims_[i], o_);
          o_ << "; ++i_" << i << "__) {" << EOL;
        }
        // add range
        generate_indent(x.dims_.size() + 2, o_);
        o_ << "param_ranges_i__.push_back(std::pair<int, int>(";
        generate_expression(x.range_.low_, o_);
        o_ << ", ";
        generate_expression(x.range_.high_, o_);
        o_ << "));" << EOL;
        // close for loop
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_indent(x.dims_.size() + 1 - i, o_);
          o_ << "}" << EOL;
        }
      }

      void operator()(const double_var_decl& x) const {
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.dims_);
      }

      void operator()(const vector_var_decl& x) const {
        generate_validate_positive(x.name_, x.M_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.M_, x.dims_);
      }

      void operator()(const row_vector_var_decl& x) const {
        generate_validate_positive(x.name_, x.N_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.N_, x.dims_);
      }

      void operator()(const matrix_var_decl& x) const {
        generate_validate_positive(x.name_, x.M_, 2, o_);
        generate_validate_positive(x.name_, x.N_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.M_, x.N_, x.dims_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += (";
        generate_expression(x.K_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }

      void operator()(const simplex_var_decl& x) const {
        // only K-1 vals
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += (";
        generate_expression(x.K_, o_);
        o_ << " - 1)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }

      void operator()(const ordered_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.K_, x.dims_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        generate_increment(x.K_, x.dims_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        generate_validate_positive(x.name_, x.M_, 2, o_);
        generate_validate_positive(x.name_, x.N_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += ((";
        // N * (N + 1) / 2  +  (M - N) * M
        generate_expression(x.N_, o_);
        o_ << " * (";
        generate_expression(x.N_, o_);
        o_ << " + 1)) / 2 + (";
        generate_expression(x.M_, o_);
        o_ << " - ";
        generate_expression(x.N_, o_);
        o_ << ") * ";
        generate_expression(x.N_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }

      void operator()(const cov_matrix_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        // (K * (K - 1))/2 + K  ?? define fun(K) = ??
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2 + ";
        generate_expression(x.K_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }

      void operator()(const corr_matrix_var_decl& x) const {
        generate_validate_positive(x.name_, x.K_, 2, o_);
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_validate_positive(x.name_, x.dims_[i], 2, o_);
        }
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
    };

  }
}
#endif
