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
      /** 
       * Vector of diagonal elements of inverse mass matrix.
       */
      Eigen::VectorXd inv_mass_matrix_;

      /**
       * Construct a diag point in n-dimensional phase space
       * with vector of ones for diagonal elements of inverse mass matrix.
       *
       * @param n number of dimensions
       */
      explicit diag_e_point(int n)
        : ps_point(n), inv_mass_matrix_(n) {
        inv_mass_matrix_.setOnes();
      }

      /**
       * Construct a diag point in n-dimensional phase space
       * with specified vector of diagonal elements of inverse mass matrix.
       *
       * @param n number of dimensions
       * @param inv_mass_matrix diagonal elements of initial mass matrix
       */
      diag_e_point(int n, Eigen::VectorXd& inv_mass_matrix)
        : ps_point(n), inv_mass_matrix_(n) {
        fast_vector_copy_<double>(inv_mass_matrix_, inv_mass_matrix);
      }

      /**
       * Copy constructor which does fast copy of inverse mass matrix.
       *
       * @param z point to copy
       */
      diag_e_point(const diag_e_point& z): ps_point(z),
                   inv_mass_matrix_(z.inv_mass_matrix_.size()) {
        fast_vector_copy_<double>(inv_mass_matrix_, z.inv_mass_matrix_);
      }

      /**
       * Write elements of mass matrix to string and handoff to writer.
       *
       * @param writer Stan writer callback
       */
      void
      write_metric(stan::callbacks::writer& writer) {
        writer("Diagonal elements of inverse mass matrix:");
        std::stringstream inv_mass_matrix_ss;
        inv_mass_matrix_ss << inv_mass_matrix_(0);
        for (int i = 1; i < inv_mass_matrix_.size(); ++i)
          inv_mass_matrix_ss << ", " << inv_mass_matrix_(i);
        writer(inv_mass_matrix_ss.str());
      }
    };

  }  // mcmc
}  // stan

#endif
