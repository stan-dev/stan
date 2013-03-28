#ifndef __STAN__AGRAD__REV__CALCULATE_CHAIN_HPP__
#define __STAN__AGRAD__REV__CALCULATE_CHAIN_HPP__

namespace stan {
  namespace agrad {
      inline double calculate_chain(const double& x, const double& val) {
        return std::exp(x - val); // works out to inv_logit(x)
      }
  }
}
#endif
