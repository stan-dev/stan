#ifndef STAN_MCMC_HMC_HAMILTONIANS_DIAG_E_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_DIAG_E_POINT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  namespace mcmc {
    /**
     * Point in a phase space with a base
     * Euclidean manifold with diagonal metric
     */
    class diag_e_point: public ps_point {
    public:
      explicit diag_e_point(int n)
        : ps_point(n), mInv(n) {
        mInv.setOnes();
      }

      Eigen::VectorXd mInv;

      diag_e_point(const diag_e_point& z): ps_point(z), mInv(z.mInv.size()) {
        fast_vector_copy_<double>(mInv, z.mInv);
      }

      void
      write_metric(stan::callbacks::writer& writer) {
        writer("Diagonal elements of inverse mass matrix:");
        std::stringstream mInv_ss;
        mInv_ss << mInv(0);
        for (int i = 1; i < mInv.size(); ++i)
          mInv_ss << ", " << mInv(i);
        writer(mInv_ss.str());
      }
    };

  }  // mcmc
}  // stan

#endif
