#ifndef STAN_VARIATIONAL_BASE_FAMILY_HPP
#define STAN_VARIATIONAL_BASE_FAMILY_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim/mat.hpp>
#include <algorithm>
#include <ostream>

namespace stan {
  namespace variational {

    class base_family {
    public:
      // Constructors
      base_family() {}

      // Operations
      base_family square() const;
      base_family sqrt() const;

      // Compound assignment operators
      base_family operator=(const base_family& rhs);
      base_family operator+=(const base_family& rhs);
      base_family operator/=(const base_family& rhs);
      base_family operator+=(double scalar);
      base_family operator*=(double scalar);

      // Distribution-based operations
      const Eigen::VectorXd& mean() const;
      double entropy() const;
      Eigen::VectorXd transform(const Eigen::VectorXd& eta) const;
      /**
       * Assign a draw from this mean field approximation to the
       * specified vector using the specified random number generator.
       *
       * @tparam BaseRNG Class of random number generator.
       * @param[in] rng Base random number generator.
       * @param[out] eta Vector to which the draw is assigned; dimension has to be
       * the same as the dimension of variational q.
       * @throws std::range_error If the index is out of range.
       */
      template <class BaseRNG>
      virtual void sample(BaseRNG& rng, Eigen::VectorXd& eta) const;
      /**
       * Draw a posterior sample from the normal distribution, 
       * and return its log normal density. The constant (d log 2 pi) is dropped. 
       *
       * @param[in] rng Base random number generator.
       * @param[out] eta Vector to which the draw is assigned; dimension has to be
       * the same as the dimension of variational q. eta will be transformed into
       * variational posteriors.
       * @param[out] log_g The log  density in the variational approximation;
       * The constant term is dropped. 
       * @throws std::range_error If the index is out of range.
       */
      template <class BaseRNG>
      virtual void sample_log_g(BaseRNG& rng, Eigen::VectorXd& eta,
                        double& log_g) const;
      /**
       * Compute the log normal density. The constant (d log 2 pi) is dropped. 
       *
       * @param[in] eta Vector; dimension has to be the same as the dimension 
       * of variational q.
       * @return The log  density in the variational approximation;
       * The constant term is dropped. 
       * @throws std::range_error If the index is out of range.
       */
      double log_p(const Eigen::VectorXd& eta) const;
      template <class M, class BaseRNG>
      void calc_grad(base_family& elbo_grad,
                     M& m,
                     Eigen::VectorXd& cont_params,
                     int n_monte_carlo_grad,
                     BaseRNG& rng,
                     callbacks::logger& logger)
        const;

    protected:
      void write_error_msg_(std::ostream* error_msgs,
                            const std::exception& e) const {
        if (!error_msgs) {
          return;
        }

        *error_msgs
          << std::endl
          << "Informational Message: The current gradient evaluation "
          << "of the ELBO is ignored because of the following issue:"
          << std::endl
          << e.what() << std::endl
          << "If this warning occurs often then your model may be "
          << "either severely ill-conditioned or misspecified."
          << std::endl;
      }
    };

    // Arithmetic operators
    base_family operator+(base_family lhs, const base_family& rhs);
    base_family operator/(base_family lhs, const base_family& rhs);
    base_family operator+(double scalar, base_family rhs);
    base_family operator*(double scalar, base_family rhs);
  }  // variational
}  // stan
#endif
