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

    template <class Model, class Point, class BaseRNG>
    class base_hamiltonian {
    public:
      explicit base_hamiltonian(Model& model)
        : model_(model), info_buffer_(), err_buffer_() {}

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
          stan::model::gradient(model_, z.q, z.V, z.g, &info_buffer_);
          z.V *= -1;
        } catch (const std::exception& e) {
          this->write_error_msg_(e);
          z.V = std::numeric_limits<double>::infinity();
        }
        z.g *= -1;
      }

      std::stringstream& info() { return info_buffer_; }
      std::stringstream& err() { return err_buffer_; }

    protected:
      Model& model_;
      std::stringstream info_buffer_;
      std::stringstream err_buffer_;

      void write_error_msg_(const std::exception& e) {
        err_buffer_
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
