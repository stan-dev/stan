#ifndef __STAN__AGRAD__AGRAD_SPECIAL_FUNCTIONS__H__
#define __STAN__AGRAD__AGRAD_SPECIAL_FUNCTIONS__H__

#include <cstddef>
#include <boost/math/special_functions.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/math/special_functions.hpp>

namespace stan {

  namespace agrad {
    
    namespace {
      
      class lgamma_vari : public op_v_vari {
      public:
        lgamma_vari(vari* avi) :
          op_v_vari(boost::math::lgamma(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * boost::math::digamma(avi_->val_);
        }
      };

      class tgamma_vari : public op_v_vari {
      public:
        tgamma_vari(vari* avi) :
          op_v_vari(boost::math::tgamma(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_ * boost::math::digamma(avi_->val_);
        }
      };

      class log1p_vari : public op_v_vari {
      public:
        log1p_vari(vari* avi) :
          op_v_vari(stan::math::log1p(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1 + avi_->val_);
        }
      };

      class log1m_vari : public op_v_vari {
      public:
        log1m_vari(vari* avi) :
          op_v_vari(stan::math::log1p(-avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (avi_->val_ - 1);
        }
      };

      class binary_log_loss_1_vari : public op_v_vari {
      public:
        binary_log_loss_1_vari(vari* avi) :
          op_v_vari(-std::log(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ / avi_->val_;
        }
      };

      class binary_log_loss_0_vari : public op_v_vari {
      public:
        binary_log_loss_0_vari(vari* avi) :
          op_v_vari(-stan::math::log1p(-avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1.0 - avi_->val_);
        }
      };

      class fdim_vv_vari : public op_vv_vari {
      public:
        fdim_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ - bvi->val_, avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_;
          bvi_->adj_ -= adj_;
        }
      };

      class fdim_vd_vari : public op_v_vari {
      public:
        fdim_vd_vari(vari* avi, double b) :
          op_v_vari(avi->val_ - b, avi) {
        }
        void chain() {
          avi_->adj_ += adj_;
        }
      };

      class fdim_dv_vari : public op_v_vari {
      public:
        fdim_dv_vari(double a, vari* bvi) :
          op_v_vari(a - bvi->val_, bvi) {
        }
        void chain() {
          // avi_ is bvi argument to constructor
          avi_->adj_ -= adj_;
        }
      };

      class fma_vvv_vari : public op_vvv_vari {
      public:
        fma_vvv_vari(vari* avi, vari* bvi, vari* cvi) :
          op_vvv_vari(avi->val_ * bvi->val_ + cvi->val_,
                      avi,bvi,cvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * bvi_->val_;
          bvi_->adj_ += adj_ * avi_->val_;
          cvi_->adj_ += adj_;
        }
      };

      class fma_vvd_vari : public op_vv_vari {
      public:
        fma_vvd_vari(vari* avi, vari* bvi, double c) :
          op_vv_vari(avi->val_ * bvi->val_ + c,
                     avi,bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * bvi_->val_;
          bvi_->adj_ += adj_ * avi_->val_;
        }
      };

      class fma_vdv_vari : public op_vdv_vari {
      public:
        fma_vdv_vari(vari* avi, double b, vari* cvi) :
          op_vdv_vari(avi->val_ * b + cvi->val_,
                      avi,b,cvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * bd_;
          cvi_->adj_ += adj_;
        }
      };

      class fma_vdd_vari : public op_vd_vari {
      public:
        fma_vdd_vari(vari* avi, double b, double c) : 
          op_vd_vari(avi->val_ * b + c,
                     avi,b) {
        }
        void chain() {
          avi_->adj_ += adj_ * bd_;
        }
      };

      class fma_ddv_vari : public op_v_vari {
      public:
        fma_ddv_vari(double a, double b, vari* cvi) :
          op_v_vari(a * b + cvi->val_, 
                    cvi) {
        }
        void chain() {
          // avi_ is cvi from constructor
          avi_->adj_ += adj_;
        }
      };

      class inv_logit_vari : public op_v_vari {
      public:
        inv_logit_vari(vari* avi) :
          op_v_vari(math::inv_logit(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ +=  adj_ * val_ * (1.0 - val_);
        }
      };

      class acosh_vari : public op_v_vari {
      public:
        acosh_vari(vari* avi) :
          op_v_vari(boost::math::acosh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / std::sqrt(avi_->val_ * avi_->val_ - 1.0);
        }
      };

      class asinh_vari : public op_v_vari {
      public:
        asinh_vari(vari* avi) :
          op_v_vari(boost::math::asinh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / std::sqrt(avi_->val_ * avi_->val_ + 1.0);
        }
      };

      class atanh_vari : public op_v_vari {
      public:
        atanh_vari(vari* avi) :
          op_v_vari(boost::math::atanh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1.0 - avi_->val_ * avi_->val_);
        }
      };

      const double TWO_OVER_SQRT_PI = 2.0 / std::sqrt(boost::math::constants::pi<double>());

      class erf_vari : public op_v_vari {
      public:
        erf_vari(vari* avi) :
          op_v_vari(boost::math::erf(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * TWO_OVER_SQRT_PI * std::exp(- avi_->val_ * avi_->val_);
        }
      };

      const double NEG_TWO_OVER_SQRT_PI = - TWO_OVER_SQRT_PI;

      class erfc_vari : public op_v_vari {
      public:
        erfc_vari(vari* avi) :
          op_v_vari(boost::math::erfc(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * NEG_TWO_OVER_SQRT_PI * std::exp(- avi_->val_ * avi_->val_);
        }
      };

      const double LOG_2 = std::log(2.0);

      class exp2_vari : public op_v_vari {
      public:
        exp2_vari(vari* avi) :
          op_v_vari(std::pow(2.0,avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_ * LOG_2;
        }
      };

      class expm1_vari : public op_v_vari {
      public:
        expm1_vari(vari* avi) :
          op_v_vari(std::exp(avi->val_) - 1.0,avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_;
        }
      };

      class hypot_vv_vari : public op_vv_vari {
      public:
        hypot_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(boost::math::hypot(avi->val_,bvi->val_),
                     avi,bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * avi_->val_ / val_;
          bvi_->adj_ += adj_ * bvi_->val_ / val_;
        }
      };

      class hypot_vd_vari : public op_v_vari {
      public:
        hypot_vd_vari(vari* avi, double b) :
          op_v_vari(boost::math::hypot(avi->val_,b),
                    avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * avi_->val_ / val_;
        }
      };

      const double LOG2 = std::log(2.0);

      class log2_vari : public op_v_vari {
      public:
        log2_vari(vari* avi) :
          op_v_vari(stan::math::log2(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (LOG2 * avi_->val_); 
        }
      };

      class cbrt_vari : public op_v_vari {
      public:
        cbrt_vari(vari* avi) :
          op_v_vari(boost::math::cbrt(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (3.0 * val_ * val_);
        }
      };

      // derivative 0 almost everywhere
      class round_vari : public vari {
      public:
        round_vari(vari* avi) :
          vari(boost::math::round(avi->val_)) {
        }
      };

      // derivative 0 almost everywhere
      class trunc_vari : public vari {
      public:
        trunc_vari(vari* avi) :
          vari(boost::math::trunc(avi->val_)) { 
        }
      };

      class inv_cloglog_vari : public op_v_vari {
      public:
        inv_cloglog_vari(vari* avi) :
          op_v_vari(stan::math::inv_cloglog(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ * std::exp(avi_->val_ - std::exp(avi_->val_));
        }
      };

      class Phi_vari : public op_v_vari {
      public:
        Phi_vari(vari* avi) :
          op_v_vari(stan::math::Phi(avi->val_), avi) {
        }
        void chain() {
          static const double NEG_HALF = -0.5;
          static const double INV_SQRT_TWO_PI 
            = 1.0 / std::sqrt(2.0 * std::sqrt(boost::math::constants::pi<double>()));
          avi_->adj_ += adj_ * INV_SQRT_TWO_PI * std::exp(NEG_HALF * avi_->val_ * avi_->val_);
        }
      };

      inline double calculate_chain(const double& x, const double& val) {
        return std::exp(x - val);
      }

      double log_sum_exp_as_double(const std::vector<var>& x) {
        using std::numeric_limits;
        using std::exp;
        using std::log;
        double max = -numeric_limits<double>::infinity();
        for (size_t i = 0; i < x.size(); ++i) 
          if (x[i] > max) 
            max = x[i].val();
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) 
          if (x[i] != -numeric_limits<double>::infinity()) 
            sum += exp(x[i].val() - max);
        return max + log(sum);
      }

      class log1p_exp_v_vari : public op_v_vari {
      public:
        log1p_exp_v_vari(vari* avi) :
          op_v_vari(stan::math::log1p_exp(avi->val_),
                    avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
        }
      };      
      class log_sum_exp_vv_vari : public op_vv_vari {
      public:
        log_sum_exp_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::log_sum_exp(avi->val_, bvi->val_),
                     avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
          bvi_->adj_ += adj_ * calculate_chain(bvi_->val_, val_);
        }
      };
      class log_sum_exp_vd_vari : public op_vd_vari {
      public:
        log_sum_exp_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::log_sum_exp(avi->val_, b),
                     avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
        }
      };
      class log_sum_exp_dv_vari : public op_dv_vari {
      public:
        log_sum_exp_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::log_sum_exp(a, bvi->val_),
                     a, bvi) {
        }
        void chain() {
          bvi_->adj_ += adj_ * calculate_chain(bvi_->val_, val_);
        }
      };
      class log_sum_exp_vector_vari : public op_vector_vari {
      public:
        log_sum_exp_vector_vari(const std::vector<var>& x) :
          op_vector_vari(log_sum_exp_as_double(x), x) {
        }
        void chain() {
          for (size_t i = 0; i < size_; ++i) {
            vis_[i]->adj_ += adj_ * calculate_chain(vis_[i]->val_, val_);
          }
        }
      };

      class square_vari : public op_v_vari {
      public:
        square_vari(vari* avi) :
          op_v_vari(avi->val_ * avi->val_,avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * 2.0 * avi_->val_;
        }
      };

      class multiply_log_vv_vari : public op_vv_vari {
      public:
        multiply_log_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::multiply_log(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          using std::log;
          avi_->adj_ += adj_ * log(bvi_->val_);
          if (bvi_->val_ == 0.0 && avi_->val_ == 0)
            bvi_->adj_ += adj_ * std::numeric_limits<double>::infinity();
          else
            bvi_->adj_ += adj_ * avi_->val_ / bvi_->val_;
        }
      };
      class multiply_log_vd_vari : public op_vd_vari {
      public:
        multiply_log_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::multiply_log(avi->val_,b),avi,b) {
        }
        void chain() {
          using std::log;
          avi_->adj_ += adj_ * log(bd_);
        }
      };
      class multiply_log_dv_vari : public op_dv_vari {
      public:
        multiply_log_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::multiply_log(a,bvi->val_),a,bvi) {
        }
        void chain() {
          if (bvi_->val_ == 0.0 && ad_ == 0.0)
            bvi_->adj_ += adj_ * std::numeric_limits<double>::infinity();
          else
            bvi_->adj_ += adj_ * ad_ / bvi_->val_;
        }
      };

    }

    /**
     * The inverse hyperbolic cosine function for variables (C99).
     * 
     * For non-variable function, see boost::math::acosh().
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \mbox{acosh}(x) = \frac{x}{x^2 - 1}\f$.
     *
     * @param a The variable.
     * @return Inverse hyperbolic cosine of the variable.
     */
    inline var acosh(const stan::agrad::var& a) {
      return var(new acosh_vari(a.vi_));
    }

    /**
     * The inverse hyperbolic sine function for variables (C99).
     * 
     * For non-variable function, see boost::math::asinh().
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \mbox{asinh}(x) = \frac{x}{x^2 + 1}\f$.
     *
     * @param a The variable.
     * @return Inverse hyperbolic sine of the variable.
     */
    inline var asinh(const stan::agrad::var& a) {
      return var(new asinh_vari(a.vi_));
    }

    /**
     * The inverse hyperbolic tangent function for variables (C99).
     *
     * For non-variable function, see boost::math::atanh().
     * 
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \mbox{atanh}(x) = \frac{1}{1 - x^2}\f$.
     *
     * @param a The variable.
     * @return Inverse hyperbolic tangent of the variable.
     */
    inline var atanh(const stan::agrad::var& a) {
      return var(new atanh_vari(a.vi_));
    }

    /**
     * The error function for variables (C99).
     *
     * For non-variable function, see boost::math::erf()
     *
     * The derivative is
     *
     * \f$\frac{d}{dx} \mbox{erf}(x) = \frac{2}{\sqrt{\pi}} \exp(-x^2)\f$.
     * 
     * @param a The variable.
     * @return Error function applied to the variable.
     */
    inline var erf(const stan::agrad::var& a) {
      return var(new erf_vari(a.vi_));
    }

    /**
     * The complementary error function for variables (C99).
     *
     * For non-variable function, see boost::math::erfc().
     *
     * The derivative is
     * 
     * \f$\frac{d}{dx} \mbox{erfc}(x) = - \frac{2}{\sqrt{\pi}} \exp(-x^2)\f$.
     *
     * @param a The variable.
     * @return Complementary error function applied to the variable.
     */
    inline var erfc(const stan::agrad::var& a) {
      return var(new erfc_vari(a.vi_));
    }

    /**
     * Exponentiation base 2 function for variables (C99).
     *
     * For non-variable function, see boost::math::exp2().
     *
     * The derivatie is
     *
     * \f$\frac{d}{dx} 2^x = (\log 2) 2^x\f$.
     * 
     * @param a The variable.
     * @return Two to the power of the specified variable.
     */
    inline var exp2(const stan::agrad::var& a) {
      return var(new exp2_vari(a.vi_));
    }

    /**
     * The exponentiation of the specified variable minus 1 (C99).
     *
     * For non-variable function, see boost::math::expm1().
     * 
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \exp(a) - 1 = \exp(a)\f$.
     * 
     * @param a The variable.
     * @return Two to the power of the specified variable.
     */
    inline var expm1(const stan::agrad::var& a) {
      return var(new expm1_vari(a.vi_));
    }

    /**
     * The log gamma function for variables (C99).  
     *
     * The derivatie is the digamma function,
     *
     * \f$\frac{d}{dx} \Gamma(x) = \psi^{(0)}(x)\f$.
     * 
     * @param a The variable.
     * @return Log gamma of the variable.
     */
    inline var lgamma(const stan::agrad::var& a) {
      return var(new lgamma_vari(a.vi_));
    }

    /**
     * The log (1 + x) function for variables (C99).
     *
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \log (1 + x) = \frac{1}{1 + x}\f$.
     *
     * @param a The variable.
     * @return The log of 1 plus the variable.
     */
    inline var log1p(const stan::agrad::var& a) {
      return var(new log1p_vari(a.vi_));
    }

    /**
     * The log (1 - x) function for variables.
     *
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \log (1 - x) = -\frac{1}{1 - x}\f$.
     *
     * @param a The variable.
     * @return The variable representing log of 1 minus the variable.
     */
    inline var log1m(const stan::agrad::var& a) {
      return var(new log1m_vari(a.vi_));
    }

    /**
     * The fused multiply-add function for three variables (C99).
     * This function returns the product of the first two arguments
     * plus the third argument.
     *
     * See boost::math::fma() for the double-based version.
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * y) + z = y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x * y) + z = x\f$, and
     *
     * \f$\frac{\partial}{\partial z} (x * y) + z = 1\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const stan::agrad::var& a,
                   const stan::agrad::var& b,
                   const stan::agrad::var& c) {
      return var(new fma_vvv_vari(a.vi_,b.vi_,c.vi_));
    }

    /**
     * The fused multiply-add function for two variables and a value
     * (C99).  This function returns the product of the first two
     * arguments plus the third argument.
     *
     * See boost::math::fma() for the double-based version.
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * y) + c = y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x * y) + c = x\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const stan::agrad::var& a,
                   const stan::agrad::var& b,
                   const double& c) {
      return var(new fma_vvd_vari(a.vi_,b.vi_,c));
    }

    /**
     * The fused multiply-add function for a variable, value, and
     * variable (C99).  This function returns the product of the first
     * two arguments plus the third argument.
     *
     * See boost::math::fma() for the double-based version.
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * c) + z = c\f$, and
     *
     * \f$\frac{\partial}{\partial z} (x * c) + z = 1\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const stan::agrad::var& a,
                   const double& b,
                   const stan::agrad::var& c) {
      return var(new fma_vdv_vari(a.vi_,b,c.vi_));
    }

    /**
     * The fused multiply-add function for a variable and two values
     * (C99).  This function returns the product of the first two
     * arguments plus the third argument.
     *
     * See boost::math::fma() for the double-based version.
     *
     * The derivative is
     *
     * \f$\frac{d}{d x} (x * c) + d = c\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const stan::agrad::var& a,
                   const double& b, 
                   const double& c) {
      return var(new fma_vdd_vari(a.vi_,b,c));
    }

    /**
     * The fused multiply-add function for a value, variable, and
     * value (C99).  This function returns the product of the first
     * two arguments plus the third argument.
     *
     * See boost::math::fma() for the double-based version.
     *
     * The derivative is
     *
     * \f$\frac{d}{d y} (c * y) + d = c\f$, and
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const double& a,
                   const stan::agrad::var& b,
                   const double& c) {
      return var(new fma_vdd_vari(b.vi_,a,c));
    }

    /**
     * The fused multiply-add function for two values and a variable,
     * and value (C99).  This function returns the product of the
     * first two arguments plus the third argument.
     *
     * See boost::math::fma() for the double-based version.
     *
     * The derivative is
     *
     * \f$\frac{\partial}{\partial z} (c * d) + z = 1\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const double& a,
                   const double& b,
                   const stan::agrad::var& c) {
      return var(new fma_ddv_vari(a,b,c.vi_));
    }

    /**
     * The fused multiply-add function for a value and two variables
     * (C99).  This function returns the product of the first two
     * arguments plus the third argument.
     *
     * See boost::math::fma() for the double-based version.
     *
     * The partial derivaties are
     *
     * \f$\frac{\partial}{\partial y} (c * y) + z = c\f$, and
     *
     * \f$\frac{\partial}{\partial z} (c * y) + z = 1\f$.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @param c Summand.
     * @return Product of the multiplicands plus the summand, ($a * $b) + $c.
     */
    inline var fma(const double& a,
                   const stan::agrad::var& b,
                   const stan::agrad::var& c) {
      return var(new fma_vdv_vari(b.vi_,a,c.vi_)); // a-b symmetry
    }

    /**
     * Returns the maximum of the two variable arguments (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * No new variable implementations are created, with this function
     * defined as if by
     *
     * <code>fmax(a,b) = a</code> if a's value is greater than b's, and .
     *
     * <code>fmax(a,b) = b</code> if b's value is greater than or equal to a's.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return If the first variable's value is larger than the
     * second's, the first variable, otherwise the second variable.
     */
    inline var fmax(const stan::agrad::var& a,
                    const stan::agrad::var& b) {
      return a.vi_->val_ > b.vi_->val_ ? a : b;
    }

    /**
     * Returns the maximum of the variable and scalar, promoting the
     * scalar to a variable if it is larger (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * For <code>fmax(a,b)</code>, if a's value is greater than b,
     * then a is returned, otherwise a fesh variable implementation
     * wrapping the value b is returned.
     *
     * @param a First variable.
     * @param b Second value
     * @return If the first variable's value is larger than or equal
     * to the second value, the first variable, otherwise the second
     * value promoted to a fresh variable.
     */
    inline var fmax(const stan::agrad::var& a,
                    const double& b) {
      return a.vi_->val_ >= b ? a : var(b);
    }

    /**
     * Returns the maximum of a scalar and variable, promoting the scalar to
     * a variable if it is larger (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * For <code>fmax(a,b)</code>, if a is greater than b's value,
     * then a fresh variable implementation wrapping a is returned, otherwise 
     * b is returned.
     *
     * @param a First value.
     * @param b Second variable.
     * @return If the first value is larger than the second variable's value,
     * return the first value promoted to a variable, otherwise return the 
     * second variable.
     */
    inline var fmax(const double& a,
                    const stan::agrad::var& b) {
      return a > b.vi_->val_ ? var(a) : b;
    }

    /**
     * Returns the minimum of the two variable arguments (C99).
     *
     * See boost::math::fmin() for the double-based version.
     *
     * For <code>fmin(a,b)</code>, if a's value is less than b's,
     * then a is returned, otherwise b is returned.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return If the first variable's value is smaller than the
     * second's, the first variable, otherwise the second variable.
     */
    inline var fmin(const stan::agrad::var& a,
                    const stan::agrad::var& b) {
      return a.vi_->val_ < b.vi_->val_ ? a : b;
    }

    /**
     * Returns the minimum of the variable and scalar, promoting the
     * scalar to a variable if it is larger (C99).
     *
     * See boost::math::fmin() for the double-based version.
     * 
     * For <code>fmin(a,b)</code>, if a's value is less than b, then a
     * is returned, otherwise a fresh variable wrapping b is returned.
     * 
     * @param a First variable.
     * @param b Second value
     * @return If the first variable's value is less than or equal to the second value,
     * the first variable, otherwise the second value promoted to a fresh variable.
     */
    inline var fmin(const stan::agrad::var& a,
                    const double& b) {
      return a.vi_->val_ <= b ? a : var(b);
    }

    /**
     * Returns the minimum of a scalar and variable, promoting the scalar to
     * a variable if it is larger (C99).
     *
     * See boost::math::fmin() for the double-based version.
     * 
     * For <code>fmin(a,b)</code>, if a is less than b's value, then a
     * fresh variable implementation wrapping a is returned, otherwise
     * b is returned.
     *
     * @param a First value.
     * @param b Second variable.
     * @return If the first value is smaller than the second variable's value,
     * return the first value promoted to a variable, otherwise return the 
     * second variable.
     */
    inline var fmin(const double& a,
                    const stan::agrad::var& b) {
      return a < b.vi_->val_ ? var(a) : b;
    }

    /**
     * Returns the length of the hypoteneuse of a right triangle
     * with sides of the specified lengths (C99).
     *
     * See boost::math::hypot() for double-based function.
     *
     * The partial derivatives are given by
     *
     * \f$\frac{\partial}{\partial x} \sqrt{x^2 + y^2} = \frac{x}{\sqrt{x^2 + y^2}}\f$, and
     *
     * \f$\frac{\partial}{\partial y} \sqrt{x^2 + y^2} = \frac{y}{\sqrt{x^2 + y^2}}\f$.
     *
     * @param a Length of first side.
     * @param b Length of second side.
     * @return Length of hypoteneuse.
     */
    inline var hypot(const stan::agrad::var& a,
                     const stan::agrad::var& b) {
      return var(new hypot_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Returns the length of the hypoteneuse of a right triangle
     * with sides of the specified lengths (C99).
     *
     * See boost::math::hypot() for double-based function.
     *
     * The derivative is
     *
     * \f$\frac{d}{d x} \sqrt{x^2 + c^2} = \frac{x}{\sqrt{x^2 + c^2}}\f$.
     * 
     * @param a Length of first side.
     * @param b Length of second side.
     * @return Length of hypoteneuse.
     */
    inline var hypot(const stan::agrad::var& a,
                     const double& b) {
      return var(new hypot_vd_vari(a.vi_,b));
    }

    /**
     * Returns the length of the hypoteneuse of a right triangle
     * with sides of the specified lengths (C99).
     *
     * See boost::math::hypot() for double-based function.
     *
     * The derivative is
     *
     * \f$\frac{d}{d y} \sqrt{c^2 + y^2} = \frac{y}{\sqrt{c^2 + y^2}}\f$.
     *
     * @param a Length of first side.
     * @param b Length of second side.
     * @return Length of hypoteneuse.
     */
    inline var hypot(const double& a,
                     const stan::agrad::var& b) {
      return var(new hypot_vd_vari(b.vi_,a));
    }

    /**
     * Returns the base 2 logarithm of the specified variable (C99).
     *
     * See stan::math::log2() for the double-based version.
     *
     * The derivative is
     *
     * \f$\frac{d}{dx} \log_2 x = \frac{1}{x \log 2}\f$.
     *
     * @param a Specified variable.
     * @return Base 2 logarithm of the variable.
     */
    inline var log2(const stan::agrad::var& a) {
      return var(new log2_vari(a.vi_));
    }

    /**
     * Returns the cube root of the specified variable (C99).
     *
     * See boost::math::cbrt() for the double-based version.
     *
     * The derivative is
     *
     * \f$\frac{d}{dx} x^{1/3} = \frac{1}{3 x^{2/3}}\f$.
     *
     * @param a Specified variable.
     * @return Cube root of the variable.
     */
    inline var cbrt(const stan::agrad::var& a) {
      return var(new cbrt_vari(a.vi_));
    }

    /**
     * Returns the rounded form of the specified variable (C99).
     *
     * See boost::math::round() for the double-based version.
     *
     * The derivative is zero everywhere but numbers half way between
     * whole numbers, so for convenience the derivative is defined to
     * be everywhere zero,
     *
     * \f$\frac{d}{dx} \mbox{round}(x) = 0\f$.
     *
     * @param a Specified variable.
     * @return Rounded variable.
     */
    inline var round(const stan::agrad::var& a) {
      return var(new round_vari(a.vi_));
    }

    /**
     * Returns the truncatation of the specified variable (C99).
     *
     * See boost::math::trunc() for the double-based version.
     *
     * The derivative is zero everywhere but at integer values, so for
     * convenience the derivative is defined to be everywhere zero,
     *
     * \f$\frac{d}{dx} \mbox{trunc}(x) = 0\f$.
     *
     * @param a Specified variable.
     * @return Truncation of the variable.
     */
    inline var trunc(const stan::agrad::var& a) {
      return var(new trunc_vari(a.vi_));
    }

    /**
     * Return the positive difference between the first variable's the value
     * and the second's (C99).
     *
     * See stan::math::fdim() for the double-based version.
     *
     * The partial derivative with respect to the first argument is
     *
     * \f$\frac{\partial}{\partial x} \mbox{fdim}(x,y) = 0.0\f$ if \f$x < y\f$, and
     *
     * \f$\frac{\partial}{\partial x} \mbox{fdim}(x,y) = 1.0\f$ if \f$x \geq y\f$.
     *
     * With respect to the second argument, the partial is
     * 
     * \f$\frac{\partial}{\partial y} \mbox{fdim}(x,y) = 0.0\f$ if \f$x < y\f$, and
     *
     * \f$\frac{\partial}{\partial y} \mbox{fdim}(x,y) = -\lfloor\frac{x}{y}\rfloor\f$ if \f$x \geq y\f$.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return The positive difference between the first and second
     * variable.
     */
    inline var fdim(const stan::agrad::var& a,
                    const stan::agrad::var& b) {
      return a.vi_->val_ > b.vi_->val_
        ? var(new fdim_vv_vari(a.vi_,b.vi_))
        : var(new vari(0.0));
    }

    /**
     * Return the positive difference between the first value and the
     * value of the second variable (C99).
     *
     * See fdim(var,var) for definitions of values and derivatives.
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d y} \mbox{fdim}(c,y) = 0.0\f$ if \f$c < y\f$, and
     *
     * \f$\frac{d}{d y} \mbox{fdim}(c,y) = -\lfloor\frac{c}{y}\rfloor\f$ if \f$c \geq y\f$.
     * 
     * @param a First value.
     * @param b Second variable.
     * @return The positive difference between the first and second
     * arguments.
     */
    inline var fdim(const double& a,
                    const stan::agrad::var& b) {
      return a > b.vi_->val_
        ? var(new fdim_dv_vari(a,b.vi_))
        : var(new vari(0.0));
    }

    /**
     * Return the positive difference between the first variable's value
     * and the second value (C99).
     *
     * See fdim(var,var) for definitions of values and derivatives.
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d x} \mbox{fdim}(x,c) = 0.0\f$ if \f$x < c\f$, and
     *
     * \f$\frac{d}{d x} \mbox{fdim}(x,c) = 1.0\f$ if \f$x \geq yc\f$.
     *
     * @param a First value.
     * @param b Second variable.
     * @return The positive difference between the first and second arguments.
     */
    inline var fdim(const stan::agrad::var& a,
                    const double& b) {
      return a.vi_->val_ > b
        ? var(new fdim_vd_vari(a.vi_,b))
        : var(new vari(0.0));
    }


    /**
     * Return the Gamma function applied to the specified variable (C99).
     *
     * See boost::math::tgamma() for the double-based version.
     *
     * The derivative with respect to the argument is
     *
     * \f$\frac{d}{dx} \Gamma(x) = \Gamma(x) \Psi^{(0)}(x)\f$
     *
     * where \f$\Psi^{(0)}(x)\f$ is the digamma function.
     *
     * See boost::math::digamma() for the double-based version.
     *
     * @param a Argument to function.
     * @return The Gamma function applied to the specified argument.
     */
    inline var tgamma(const stan::agrad::var& a) {
      return var(new tgamma_vari(a.vi_));
    }

    /**
     * Return the step, or heaviside, function applied to the
     * specified variable (stan).
     *
     * See stan::math::step() for the double-based version.
     *
     * The derivative of the step function is zero everywhere
     * but at 0, so for convenience, it is taken to be everywhere
     * zero,
     *
     * \f$\mbox{step}(x) = 0\f$.
     *
     * @param a Variable argument.
     * @return The constant variable with value 1.0 if the argument's
     * value is greater than or equal to 0.0, and value 0.0 otherwise.
     */
    inline var step(const stan::agrad::var& a) {
      return var(new vari(a.vi_->val_ < 0.0 ? 0.0 : 1.0));
    }

    /**
     * Return the inverse complementary log-log function applied
     * specified variable (stan).
     *
     * See stan::math::inv_cloglog() for the double-based version.
     *
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \mbox{cloglog}^{-1}(x) = \exp (x - \exp (x))\f$.
     *
     * @param a Variable argument.
     * @return The inverse complementary log-log of the specified
     * argument.
     */
    inline var inv_cloglog(const stan::agrad::var& a) {
      return var(new inv_cloglog_vari(a.vi_));
    }

    /**
     * The unit normal cumulative density function for variables (stan).
     *
     * See stan::math::Phi() for the double-based version.
     *
     * The derivative is the unit normal density function,
     *
     * \f$\frac{d}{dx} \Phi(x) = \mbox{\sf Norm}(x|0,1) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{1}{2} x^2)\f$.
     *
     * @param a Variable argument.
     * @return The unit normal cdf evaluated at the specified argument.
     */
    inline var Phi(const stan::agrad::var& a) {
      return var(new Phi_vari(a.vi_));
    }

    /**
     * The inverse logit function for variables (stan).
     *
     * See stan::math::inv_logit() for the double-based version.
     *
     * The derivative of inverse logit is
     *
     * \f$\frac{d}{dx} \mbox{logit}^{-1}(x) = \mbox{logit}^{-1}(x) (1 - \mbox{logit}^{-1}(x))\f$.
     *
     * @param a Argument variable.
     * @return Inverse logit of argument.
     */
    inline var inv_logit(const stan::agrad::var& a) {
      return var(new inv_logit_vari(a.vi_));
    }

    /**
     * The log loss function for variables (stan).
     *
     * See stan::math::log_loss() for the double-based version.
     *
     * The derivative with respect to the variable \f$\hat{y}\f$ is
     *
     * \f$\frac{d}{d\hat{y}} \mbox{logloss}(1,\hat{y}) = - \frac{1}{\hat{y}}\f$, and
     *
     * \f$\frac{d}{d\hat{y}} \mbox{logloss}(0,\hat{y}) = \frac{1}{1 - \hat{y}}\f$.
     *
     * @param y Reference value.
     * @param y_hat Response variable.
     * @return Log loss of response versus reference value.
     */
    inline var log_loss(const int& y, 
                        const stan::agrad::var& y_hat) {
      return y == 0  
        ? var(new binary_log_loss_0_vari(y_hat.vi_))
        : var(new binary_log_loss_1_vari(y_hat.vi_));
    }
    
    /**
     * Return the log of 1 plus the exponential of the specified
     * variable.
     */
    inline var log1p_exp(const stan::agrad::var& a) {
      return var(new log1p_exp_v_vari(a.vi_));
    }

    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const stan::agrad::var& a,
                           const stan::agrad::var& b) {
      return var(new log_sum_exp_vv_vari(a.vi_, b.vi_));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const stan::agrad::var& a,
                           const double& b) {
      return var(new log_sum_exp_vd_vari(a.vi_, b));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const double& a,
                           const stan::agrad::var& b) {
      return var(new log_sum_exp_dv_vari(a, b.vi_));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const std::vector<var>& x) {
      return var(new log_sum_exp_vector_vari(x));
    }

    /**
     * Return the square of the input variable.
     *
     * <p>Using <code>square(x)</code> is more efficient
     * than using <code>x * x</code>.
     *
     * @param x Variable to square.
     * @return Square of variable.
     */
    inline var square(const var& x) {
      return var(new square_vari(x.vi_));
    }

    // OTHER FUNCTIONS: stan/math/special_functions.hpp implementations
    /**
     * Return the value of a*log(b).
     *
     * When both a and b are 0, the value returned is 0.
     * The partial deriviative with respect to a is log(b). 
     * The partial deriviative with respect to b is a/b. When
     * a and b are both 0, this is set to Inf.
     *
     * @param a First variable.
     * @param b Second variable.
     * @return Value of a*log(b)
     */
    inline var multiply_log(const var& a, const var& b) {
      return var(new multiply_log_vv_vari(a.vi_,b.vi_));
    }
    /**
     * Return the value of a*log(b).
     *
     * When both a and b are 0, the value returned is 0.
     * The partial deriviative with respect to a is log(b). 
     *
     * @param a First variable.
     * @param b Second scalar.
     * @return Value of a*log(b)
     */
    inline var multiply_log(const var& a, const double b) {
      return var(new multiply_log_vd_vari(a.vi_,b));
    }
    /**
     * Return the value of a*log(b).
     *
     * When both a and b are 0, the value returned is 0.
     * The partial deriviative with respect to b is a/b. When
     * a and b are both 0, this is set to Inf.
     *
     * @param a First scalar.
     * @param b Second variable.
     * @return Value of a*log(b)
     */
    inline var multiply_log(const double a, const var& b) {
      if (a == 1.0)
        return log(b);
      return var(new multiply_log_dv_vari(a,b.vi_));
    }


    /**
     * If the specified condition is true, return the first
     * variable, otherwise return the second variable.
     *
     * @param c Boolean condition.
     * @param y_true Variable to return if condition is true.
     * @param y_false Variable to return if condition is false.
     */
    inline var if_else(bool c, const var& y_true, const var&y_false) {
      return c ? y_true : y_false;
    }
    /**
     * If the specified condition is true, return a new variable
     * constructed from the first scalar, otherwise return the second
     * variable.
     *
     * @param c Boolean condition.
     * @param y_true Value to promote to variable and return if condition is true.
     * @param y_false Variable to return if condition is false.
     */
    inline var if_else(bool c, double y_true, const var& y_false) {
      if (c) 
        return var(y_true);
      else 
        return y_false;
    }
    /**
     * If the specified condition is true, return the first variable,
     * otherwise return a new variable constructed from the second
     * scalar.
     *
     * @param c Boolean condition.
     * @param y_true Variable to return if condition is true.
     * @param y_false Value to promote to variable and return if condition is false.
     */
    inline var if_else(bool c, const var& y_true, const double y_false) {
      if (c) 
        return y_true;
      else 
        return var(y_false);
    }

  }
}

#endif
