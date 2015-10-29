#ifndef STAN_MCMC_HMC_HAMILTONIANS_BASE_HAMILTONIAN_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_BASE_HAMILTONIAN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/model/util.hpp>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stan {

  namespace mcmc {

    template <typename Model, typename Point, typename BaseRNG>
    class base_hamiltonian {
    public:
      base_hamiltonian(Model& m, std::ostream* e)
        : model_(m), err_stream_(e) {}

      ~base_hamiltonian() {}

      typedef Point PointType;

      virtual double T(Point& z) = 0;

      double V(Point& z) {
        return z.V;
      }

      virtual double tau(Point& z) = 0;

      virtual double phi(Point& z) = 0;

      double H(Point& z) {
        return T(z) + V(z);
      }

      // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q)
      virtual const Eigen::VectorXd dtau_dq(Point& z) = 0;

      virtual const Eigen::VectorXd dtau_dp(Point& z) = 0;

      // phi = 0.5 * log | Lambda (q) | + V(q)
      virtual const Eigen::VectorXd dphi_dq(Point& z) = 0;

      virtual void sample_p(Point& z, BaseRNG& rng) = 0;

      virtual void init(Point& z) {
        this->update(z);
      }

      virtual void update(Point& z) {
        try {
          stan::model::gradient(model_, z.q, z.V, z.g, err_stream_);
          z.V *= -1;
        } catch (const std::exception& e) {
          this->write_error_msg_(err_stream_, e);
          z.V = std::numeric_limits<double>::infinity();
        }
        z.g *= -1;
      }

    protected:
        Model& model_;

        std::ostream* err_stream_;

        void write_error_msg_(std::ostream* error_msgs,
                              const std::exception& e) {
          if (!error_msgs)
            return;

          *error_msgs
            << std::endl
            << "Informational Message: The current Metropolis proposal "
            << "is about to be rejected because of the following issue:"
            << std::endl
            << e.what() << std::endl
            << "If this warning occurs sporadically, such as for highly "
            << "constrained variable types like covariance matrices, then "
            << "the sampler is fine,"
            << std::endl
            << "but if this warning occurs often then your model may be "
            << "either severely ill-conditioned or misspecified."
            << std::endl;
      }
    };

  }  // mcmc
}  // stan

#endif
