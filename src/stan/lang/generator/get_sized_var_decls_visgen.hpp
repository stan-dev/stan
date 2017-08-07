#ifndef STAN_LANG_GENERATOR_GET_SIZED_VAR_DECLS_VISGEN_HPP
#define STAN_LANG_GENERATOR_GET_SIZED_VAR_DECLS_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Visitor for generating code to push static variable
     * declarations onto an accumulator.  return static variable
     */
    struct get_sized_var_decls_visgen : public visgen {
      /**
       * Construct a get variable declarations visitor for the
       * specified stream.  This one is for the runtime method that
       * returns sizes based on expressions in the declaration.
       *
       * @param[in,out] o stream for generating
       */
      get_sized_var_decls_visgen(std::ostream& o)
        : visgen(3, o) { }

      void push_back_var_decl(const std::string& type_name,
                              const std::string& name,
                              const std::vector<expression>& dims) const {
        o_ << INDENT3 << "decls__.push_back(sized_var_decl("
           << "\"" << name << "\", "
           << "\"" << type_name << "\", "
           << dims.size() << ", "
           << "stan::math::array_builder<int>()";
        // fill out dims as array builder argument here
        o_ << ".array()));" << std::endl;
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        push_back_var_decl("int", x.name_, x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        push_back_var_decl("real", x.name_, x.dims_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        push_back_var_decl("unit_vector", x.name_, x.dims_);
      }

      void operator()(const simplex_var_decl& x) const {
        push_back_var_decl("simplex", x.name_, x.dims_);
      }

      void operator()(const ordered_var_decl& x) const {
        push_back_var_decl("ordered", x.name_, x.dims_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        push_back_var_decl("positive_ordered", x.name_, x.dims_);
      }

      void operator()(const vector_var_decl& x) const {
        push_back_var_decl("vector", x.name_, x.dims_);
      }

      void operator()(const row_vector_var_decl& x) const {
        push_back_var_decl("row_vector", x.name_, x.dims_);
      }

      void operator()(const matrix_var_decl& x) const {
        push_back_var_decl("matrix", x.name_, x.dims_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        push_back_var_decl("cholesky_factor_cov", x.name_, x.dims_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        push_back_var_decl("cholesky_factor_corr", x.name_, x.dims_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        push_back_var_decl("cov_matrix", x.name_, x.dims_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        push_back_var_decl("corr_matrix", x.name_, x.dims_);
      }
    };

  }
}
#endif
