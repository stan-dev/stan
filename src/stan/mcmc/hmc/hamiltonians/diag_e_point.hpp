#ifndef STAN__MCMC__DIAG__E__POINT__BETA
#define STAN__MCMC__DIAG__E__POINT__BETA

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {

  namespace mcmc {

    // Point in a phase space with a base
    // Euclidean manifold with diagonal metric
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

      template <class Writer>
      void write_metric(Writer& writer) {
        writer("# Diagonal Euclidean metric");
        writer("M_inv", mInv.data(), mInv.size());
      }
    };

  }  // mcmc

}  // stan

#endif
