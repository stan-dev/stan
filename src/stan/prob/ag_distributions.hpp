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
      double sigmasq_inv_, log_sigma_;
      std::vector<double> y_minus_mu_;
    public:
      normal_log_vec_vvv_vari(std::vector<agrad::vari*> y, agrad::vari* mu, agrad::vari* sigma) :
        agrad::vari(0), y_(y), mu_(mu), sigma_(sigma), 
        y_minus_mu_(y.size()) {
        double& val = const_cast<double&>(val_);
        sigmasq_inv_ = 0.5 / (sigma_->val_ * sigma_->val_);
        log_sigma_ = std::log(sigma_->val_);
        for (unsigned int i = 0; i < y.size(); i++) {
          y_minus_mu_[i] = y_[i]->val_ - mu->val_;
          val -= 0.5 * sigmasq_inv_ * y_minus_mu_[i] * y_minus_mu_[i];
        }
        val -= y.size() * (log_sigma_ - NEG_LOG_SQRT_TWO_PI);
      }

      void chain() {
        double sigmacubed_inv2 = 2 * sigmasq_inv_ / sigma_->val_;
        for (unsigned int i = 0; i < y_.size(); i++) {
          y_[i]->adj_ -= adj_ * y_minus_mu_[i] * sigmasq_inv_;
          mu_->adj_ += adj_ * y_minus_mu_[i] * sigmasq_inv_;
          sigma_->adj_ += adj_ * 0.5 * y_minus_mu_[i] * y_minus_mu_[i] * 
            sigmacubed_inv2;
        }
        sigma_->adj_ -= adj_ * y_.size() * 1.0 / sigma_->val_;
      }
    };

    class normal_log_vec_dvv_vari : public agrad::vari {
    protected:
      agrad::vari* mu_;
      agrad::vari* sigma_;
      double sigmasq_inv_, log_sigma_;
      std::vector<double> y_minus_mu_;
    public:
      normal_log_vec_dvv_vari(std::vector<double> y, agrad::vari* mu, agrad::vari* sigma) :
        agrad::vari(0), mu_(mu), sigma_(sigma), 
        y_minus_mu_(y.size()) {
        double& val = const_cast<double&>(val_);
        sigmasq_inv_ = 0.5 / (sigma_->val_ * sigma_->val_);
        log_sigma_ = std::log(sigma_->val_);
        for (unsigned int i = 0; i < y.size(); i++) {
          y_minus_mu_[i] = y[i] - mu->val_;
          val -= 0.5 * sigmasq_inv_ * y_minus_mu_[i] * y_minus_mu_[i];
        }
        val -= y.size() * (log_sigma_ - NEG_LOG_SQRT_TWO_PI);
      }

      void chain() {
        double sigmacubed_inv2 = 2 * sigmasq_inv_ / sigma_->val_;
        for (unsigned int i = 0; i < y_minus_mu_.size(); i++) {
          mu_->adj_ += adj_ * y_minus_mu_[i] * sigmasq_inv_;
          sigma_->adj_ += adj_ * 0.5 * y_minus_mu_[i] * y_minus_mu_[i] * 
            sigmacubed_inv2;
        }
        sigma_->adj_ -= adj_ * y_minus_mu_.size() * 1.0 / sigma_->val_;
      }
    };

    class normal_log_dvv_vari : public agrad::vari {
    protected:
      agrad::vari* mu_;
      agrad::vari* sigma_;
      double y_minus_mu_, y_minus_mu_sigmasq_inv_;
    public:
      normal_log_dvv_vari(double y, agrad::vari* mu, agrad::vari* sigma) :
        agrad::vari(0), mu_(mu), sigma_(sigma), 
        y_minus_mu_(y-mu->val_),
        y_minus_mu_sigmasq_inv_(y_minus_mu_ / (sigma_->val_ * sigma_->val_)) {
        const_cast<double&>(val_) = -0.5 * y_minus_mu_sigmasq_inv_ * y_minus_mu_
          - log(sigma->val_) - NEG_LOG_SQRT_TWO_PI;
      }
      
      void chain() {
        mu_->adj_ += adj_ * y_minus_mu_sigmasq_inv_;
        sigma_->adj_ += adj_ * (y_minus_mu_sigmasq_inv_ * y_minus_mu_ / 
                                sigma_->val_ - 1.0 / sigma_->val_);
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
