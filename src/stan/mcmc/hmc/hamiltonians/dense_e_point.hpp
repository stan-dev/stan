#ifndef STAN_MCMC_HMC_HAMILTONIANS_DENSE_E_POINT_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_DENSE_E_POINT_HPP

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {

  namespace mcmc {

    // Point in a phase space with a base
    // Euclidean manifold with dense metric
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

      /**
       * @tparam Writer An implementation of
       *                    src/stan/interface_callbacks/writer/base_writer.hpp
       * @param writer Writer callback
       */
      template <class Writer>
      void write_metric(Writer& writer) {
        writer("# Dense Euclidean metric");
        writer("# M_inv", mInv.data(), mInv.rows(), mInv.cols());
      }
    };

  }  // mcmc

}  // stan

#endif
