#ifndef STAN_MCMC_HMC_HAMILTONIANS_DENSE_E_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_DENSE_E_POINT_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  namespace mcmc {
    /**
     * Point in a phase space with a base
     * Euclidean manifold with dense metric
     */
    class dense_e_point: public ps_point {
    public:
      explicit dense_e_point(int n)
        : ps_point(n), mInv(n, n) {
        mInv.setIdentity();
      }

      Eigen::MatrixXd mInv;

      dense_e_point(const dense_e_point& z)
        : ps_point(z), mInv(z.mInv.rows(), z.mInv.cols()) {
        fast_matrix_copy_<double>(mInv, z.mInv);
      }

      void
      write_metric(stan::callbacks::writer& writer) {
        writer("Elements of inverse mass matrix:");
        for (int i = 0; i < mInv.rows(); ++i) {
          std::stringstream mInv_ss;
          mInv_ss << mInv(i, 0);
          for (int j = 1; j < mInv.cols(); ++j)
            mInv_ss << ", " << mInv(i, j);
          writer(mInv_ss.str());
        }
      }
    };

  }  // mcmc
}  // stan

#endif
