#ifndef STAN_LANG_GENERATOR_GENERATE_EIGEN_INDEX_EXPRESSION_HPP
#define STAN_LANG_GENERATOR_GENERATE_EIGEN_INDEX_EXPRESSION_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate the specified expression cast to an Eigen index to
     * disambiguate from pointer.
     *
     * @param[in] e expression for size
     * @param[in,out] o stream for generating
     */
    // use to disambiguate VectorXd(0) ctor from Scalar* alternative
    void generate_eigen_index_expression(const expression& e, std::ostream& o) {
      o << "static_cast<Eigen::VectorXd::Index>(";
      generate_expression(e.expr_, NOT_USER_FACING, o);
      o << ")";
    }



  }
}
#endif
