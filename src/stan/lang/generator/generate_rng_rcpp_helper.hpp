#ifndef STAN_LANG_GENERATOR_GENERATE_RNG_RCPP_HELPER_HPP
#define STAN_LANG_GENERATOR_GENERATE_RNG_RCPP_HELPER_HPP

#include <stan/lang/generator/constants.hpp>
#include <ostream>

namespace stan {
  namespace lang {

   /**
     * Generate a helper function to export RNG instances to RCpp.
     * This should match the default RNG chosen in 
     * generate_function_instantiation()
     *
     * @param[in] o output stream
     */
    void generate_rng_rcpp_helper(std::ostream& o) {
      o << "// [[Rcpp::export]]" << EOL;
      o << "boost::ecuyer1988 __create_rng(int seed) {" << EOL;
      o << "  return(boost::ecuyer1988(seed));" << EOL;
      o << "}" << EOL << EOL;
    }

  }
}
#endif
