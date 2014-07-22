#ifndef __STAN__VB__LATENT_VARS__HPP__
#define __STAN__VB__LATENT_VARS__HPP__

#include <vector>

#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace vb {

    class latent_vars
    {

    private:

      Eigen::VectorXd mu_; // Mean of location-scale family
      Eigen::MatrixXd L_;  // Lower-triangular decomposition of scale matrix
      int dimension_;

    public:

      latent_vars(Eigen::VectorXd const& mu, Eigen::MatrixXd const& L) :
      mu_(mu), L_(L), dimension_(mu.size())
      {

        if (dimension_ != L_.rows() || dimension_ != L_.cols())
          throw std::runtime_error("[latent_vars] mu and L "
                                  "dimensions do not match.");

      };

      virtual ~latent_vars() {}; // No-op

      int dimension() const
      {
        return dimension_;
      }

      Eigen::VectorXd const& mu() const
      {
        return mu_;
      }

      Eigen::MatrixXd const& L() const
      {
        return L_;
      }

      // Implements g^{-1}(\check{z}) = L\check{z} + \mu
      Eigen::VectorXd to_unconstrained(Eigen::VectorXd const& x)
      {
        if (mu_.size() != x.size())
          throw std::runtime_error("[latent_vars::to_unconstrained] input "
                             "dimension does not match internal dimension.");

        return L_*x + mu_;
      }

      // Implements g(\widetilde{z}) = L^{-1}(\check{z} - \mu)
      Eigen::VectorXd to_standardized(Eigen::VectorXd const& x)
      {
        if (mu_.size() != x.size())
          throw std::runtime_error("[latent_vars::to_standardized] input "
                             "dimension does not match internal dimension.");

        return L_.partialPivLu().solve( x - mu_ );
      }

    };

  } // vb

} // stan

#endif
