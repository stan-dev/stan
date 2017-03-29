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
      /** 
       * Inverse mass matrix.
       */
      Eigen::MatrixXd inv_mass_matrix_;

      /**
       * Construct a dense point in n-dimensional phase space
       * with identity matrix as inverse mass matrix.
       *
       * @param n number of dimensions
       */
      explicit dense_e_point(int n)
        : ps_point(n), inv_mass_matrix_(n, n) {
        inv_mass_matrix_.setIdentity();
      }

      /**
       * Construct a dense point in n-dimensional phase space
       * with specified inverse mass matrix.
       *
       * @param n number of dimensions
       * @param inv_mass_matrix initial mass matrix
       */
      dense_e_point(int n, Eigen::MatrixXd inv_mass_matrix)
        : ps_point(n), inv_mass_matrix_(n, n) {
        fast_matrix_copy_<double>(inv_mass_matrix_, inv_mass_matrix);
      }

      /**
       * Copy constructor which does fast copy of inverse mass matrix.
       *
       * @param z point to copy
       */
      dense_e_point(const dense_e_point& z)
        : ps_point(z), inv_mass_matrix_(z.inv_mass_matrix_.rows(),
                                            z.inv_mass_matrix_.cols()) {
        fast_matrix_copy_<double>(inv_mass_matrix_, z.inv_mass_matrix_);
      }

      /**
       * Write elements of mass matrix to string and handoff to writer.
       *
       * @param writer Stan writer callback
       */
      void
      write_metric(stan::callbacks::writer& writer) {
        writer("Elements of inverse mass matrix:");
        for (int i = 0; i < inv_mass_matrix_.rows(); ++i) {
          std::stringstream inv_mass_matrix_ss;
          inv_mass_matrix_ss << inv_mass_matrix_(i, 0);
          for (int j = 1; j < inv_mass_matrix_.cols(); ++j)
            inv_mass_matrix_ss << ", " << inv_mass_matrix_(i, j);
          writer(inv_mass_matrix_ss.str());
        }
      }
    };

  }  // mcmc
}  // stan

#endif
