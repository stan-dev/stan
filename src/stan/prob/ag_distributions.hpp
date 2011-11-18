#ifndef __STAN_PROB_AG_DISTRIBUTIONS__HPP__
#define __STAN_PROB_AG_DISTRIBUTIONS__HPP__

#include <stan/agrad/agrad.hpp>
#include <stan/prob/distributions.hpp>

namespace stan {

  namespace prob {
    
    class normal_log_vec_vvv_vari : public agrad::vari {
    protected:
      std::vector<agrad::vari*> y_;
      agrad::vari* mu_;
      agrad::vari* sigma_;
      double mu_adj2_, sigma_adj2_;
      std::vector<double> y_adj2_;
    public:
      normal_log_vec_vvv_vari(std::vector<agrad::vari*>& y, 
                              agrad::vari* mu, agrad::vari* sigma) :
        agrad::vari(0), y_(y), mu_(mu), sigma_(sigma), mu_adj2_(0), 
        sigma_adj2_(0), y_adj2_(y.size()) {
        double& val = const_cast<double&>(val_);
        double sigma_inv = 1.0 / sigma_->val_;
        double sigmasq_inv = sigma_inv * sigma_inv;
        double sigmacubed_inv = sigmasq_inv * sigma_inv;
        double half_sigmasq_inv = 0.5 * sigmasq_inv;
        double log_sigma = std::log(sigma_->val_);
        for (unsigned int i = 0; i < y.size(); ++i) {
          double y_minus_mu = y_[i]->val_ - mu->val_;
          double y_minus_mu_sq = y_minus_mu * y_minus_mu;
          val -= half_sigmasq_inv * y_minus_mu_sq;
          sigma_adj2_ += sigmacubed_inv * y_minus_mu_sq;
          mu_adj2_ += sigmasq_inv * y_minus_mu;
          y_adj2_[i] -= sigmasq_inv * y_minus_mu;
        }
        val -= y.size() * (log_sigma - NEG_LOG_SQRT_TWO_PI);
        sigma_adj2_ -= y.size() * sigma_inv;
      }

      void chain() {
        for (unsigned int i = 0; i < y_.size(); ++i)
          y_[i]->adj_ += adj_ * y_adj2_[i];
        mu_->adj_ += adj_ * mu_adj2_;
        sigma_->adj_ += adj_ * sigma_adj2_;
      }
    };
    
    class normal_log_vec_dvv_vari : public agrad::vari {
    protected:
      agrad::vari* mu_;
      agrad::vari* sigma_;
      double mu_adj2_, sigma_adj2_;
    public:
      normal_log_vec_dvv_vari(const std::vector<double>& y, agrad::vari* mu,
                              agrad::vari* sigma) :
        agrad::vari(0), mu_(mu), sigma_(sigma), mu_adj2_(0), 
        sigma_adj2_(0) {
        double& val = const_cast<double&>(val_);
        double sigma_inv = 1.0 / sigma_->val_;
        double sigmasq_inv = sigma_inv * sigma_inv;
        double sigmacubed_inv = sigmasq_inv * sigma_inv;
        double half_sigmasq_inv = 0.5 * sigmasq_inv;
        double log_sigma = std::log(sigma_->val_);
        for (unsigned int i = 0; i < y.size(); ++i) {
          double y_minus_mu = y[i] - mu->val_;
          double y_minus_mu_sq = y_minus_mu * y_minus_mu;
          val -= half_sigmasq_inv * y_minus_mu_sq;
          sigma_adj2_ += sigmacubed_inv * y_minus_mu_sq;
          mu_adj2_ += sigmasq_inv * y_minus_mu;
        }
        val -= y.size() * (log_sigma - NEG_LOG_SQRT_TWO_PI);
        sigma_adj2_ -= y.size() * sigma_inv;
      }

      void chain() {
        mu_->adj_ += adj_ * mu_adj2_;
        sigma_->adj_ += adj_ * sigma_adj2_;
      }
    };

    class normal_log_dvv_vari : public agrad::vari {
    protected:
      agrad::vari* mu_;
      agrad::vari* sigma_;
      double mu_adj2_, sigma_adj2_;
    public:
      normal_log_dvv_vari(double y, agrad::vari* mu, agrad::vari* sigma) :
        agrad::vari(0), mu_(mu), sigma_(sigma), mu_adj2_(0), sigma_adj2_(0) {
        double sigma_inv = 1.0 / sigma->val_;
        double y_minus_mu = y - mu->val_;
        double sigmasq_inv_y_minus_mu = sigma_inv * sigma_inv * y_minus_mu;
        double sigmasq_inv_y_minus_mu_sq = sigmasq_inv_y_minus_mu * y_minus_mu;
        const_cast<double&>(val_) = -0.5 * sigmasq_inv_y_minus_mu_sq
          - log(sigma->val_) - NEG_LOG_SQRT_TWO_PI;
        mu_adj2_ = sigmasq_inv_y_minus_mu;
        sigma_adj2_ = (sigmasq_inv_y_minus_mu_sq - 1) * sigma_inv;
      }
      
      void chain() {
        mu_->adj_ += adj_ * mu_adj2_;
        sigma_->adj_ += adj_ * sigma_adj2_;
      }
    };

    template<>
    inline agrad::var normal_log<double, agrad::var, agrad::var>
    (const double& y, const agrad::var& mu, const agrad::var& sigma) {
      return agrad::var(new normal_log_dvv_vari(y, mu.vi_, sigma.vi_));
    }

    template<>
    inline agrad::var normal_log<agrad::var, agrad::var, agrad::var>
    (const std::vector<agrad::var>& y, const agrad::var& mu, const agrad::var& sigma) {
      std::vector<agrad::vari*> yvari(y.size());
      for (unsigned int i = 0; i < y.size(); i++) { yvari[i] = y[i].vi_; }
      return agrad::var(new normal_log_vec_vvv_vari(yvari, mu.vi_, sigma.vi_));
    }

    template<>
    inline agrad::var normal_log<double, agrad::var, agrad::var>
    (const std::vector<double>& y, const agrad::var& mu, const agrad::var& sigma) {
      return agrad::var(new normal_log_vec_dvv_vari(y, mu.vi_, sigma.vi_));
    }

  }

}

#endif
