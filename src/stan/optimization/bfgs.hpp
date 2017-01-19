#ifndef STAN_OPTIMIZATION_BFGS_HPP
#define STAN_OPTIMIZATION_BFGS_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/model/log_prob_propto.hpp>
#include <stan/model/log_prob_grad.hpp>
#include <stan/optimization/bfgs_linesearch.hpp>
#include <stan/optimization/bfgs_update.hpp>
#include <stan/optimization/lbfgs_update.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace stan {
  namespace optimization {
    typedef enum {
      TERM_SUCCESS = 0,
      TERM_ABSX = 10,
      TERM_ABSF = 20,
      TERM_RELF = 21,
      TERM_ABSGRAD = 30,
      TERM_RELGRAD = 31,
      TERM_MAXIT = 40,
      TERM_LSFAIL = -1
    } TerminationCondition;

    template<typename Scalar = double>
    class ConvergenceOptions {
    public:
      ConvergenceOptions() {
        maxIts = 10000;
        fScale = 1.0;

        tolAbsX = 1e-8;
        tolAbsF = 1e-12;
        tolAbsGrad = 1e-8;

        tolRelF = 1e+4;
        tolRelGrad = 1e+3;
      }
      size_t maxIts;
      Scalar tolAbsX;
      Scalar tolAbsF;
      Scalar tolRelF;
      Scalar fScale;
      Scalar tolAbsGrad;
      Scalar tolRelGrad;
    };

    template<typename Scalar = double>
    class LSOptions {
    public:
      LSOptions() {
        c1 = 1e-4;
        c2 = 0.9;
        minAlpha = 1e-12;
        alpha0 = 1e-3;
      }
      Scalar c1;
      Scalar c2;
      Scalar alpha0;
      Scalar minAlpha;
    };
    template<typename FunctorType, typename QNUpdateType,
             typename Scalar = double, int DimAtCompile = Eigen::Dynamic>
    class BFGSMinimizer {
    public:
      typedef Eigen::Matrix<Scalar, DimAtCompile, 1> VectorT;
      typedef Eigen::Matrix<Scalar, DimAtCompile, DimAtCompile> HessianT;

    protected:
      FunctorType &func_;
      VectorT gk_, gk_1_, xk_1_, xk_, pk_, pk_1_;
      Scalar fk_, fk_1_, alphak_1_;
      Scalar alpha_, alpha0_;
      size_t itNum_;
      std::string note_;
      QNUpdateType qn_;

    public:
      LSOptions<Scalar> ls_opts_;
      ConvergenceOptions<Scalar> conv_opts_;

      QNUpdateType &get_qnupdate() { return qn_; }
      const QNUpdateType &get_qnupdate() const { return qn_; }

      const Scalar &curr_f() const { return fk_; }
      const VectorT &curr_x() const { return xk_; }
      const VectorT &curr_g() const { return gk_; }
      const VectorT &curr_p() const { return pk_; }

      const Scalar &prev_f() const { return fk_1_; }
      const VectorT &prev_x() const { return xk_1_; }
      const VectorT &prev_g() const { return gk_1_; }
      const VectorT &prev_p() const { return pk_1_; }
      Scalar prev_step_size() const { return pk_1_.norm()*alphak_1_; }

      inline Scalar rel_grad_norm() const {
        return -pk_.dot(gk_) / std::max(std::fabs(fk_), conv_opts_.fScale);
      }
      inline Scalar rel_obj_decrease() const {
        return std::fabs(fk_1_ - fk_) / std::max(std::fabs(fk_1_),
                                                 std::max(std::fabs(fk_),
                                                          conv_opts_.fScale));
      }

      const Scalar &alpha0() const { return alpha0_; }
      const Scalar &alpha() const { return alpha_; }
      const size_t iter_num() const { return itNum_; }

      const std::string &note() const { return note_; }

      std::string get_code_string(int retCode) {
        switch (retCode) {
          case TERM_SUCCESS:
            return std::string("Successful step completed");
          case TERM_ABSF:
            return std::string("Convergence detected: absolute change "
                               "in objective function was below tolerance");
          case TERM_RELF:
            return std::string("Convergence detected: relative change "
                               "in objective function was below tolerance");
          case TERM_ABSGRAD:
            return std::string("Convergence detected: "
                               "gradient norm is below tolerance");
          case TERM_RELGRAD:
            return std::string("Convergence detected: relative "
                               "gradient magnitude is below tolerance");
          case TERM_ABSX:
            return std::string("Convergence detected: "
                               "absolute parameter change was below tolerance");
          case TERM_MAXIT:
            return std::string("Maximum number of iterations hit, "
                               "may not be at an optima");
          case TERM_LSFAIL:
            return std::string("Line search failed to achieve a sufficient "
                               "decrease, no more progress can be made");
          default:
            return std::string("Unknown termination code");
        }
      }

      explicit BFGSMinimizer(FunctorType &f) : func_(f) { }

      void initialize(const VectorT &x0) {
        int ret;
        xk_ = x0;
        ret = func_(xk_, fk_, gk_);
        if (ret) {
          throw std::runtime_error("Error evaluating initial BFGS point.");
        }
        pk_ = -gk_;

        itNum_ = 0;
        note_ = "";
      }

      int step() {
        Scalar gradNorm, stepNorm;
        VectorT sk, yk;
        int retCode(0);
        int resetB(0);

        itNum_++;

        if (itNum_ == 1) {
          resetB = 1;
          note_ = "";
        } else {
          resetB = 0;
          note_ = "";
        }

        while (true) {
          if (resetB) {
            // Reset the Hessian approximation
            pk_.noalias() = -gk_;
          }

          // Get an initial guess for the step size (alpha)
          if (itNum_ > 1 && resetB != 2) {
            // use cubic interpolation based on the previous step
            alpha0_ = alpha_ = std::min(1.0,
                                        1.01*CubicInterp(gk_1_.dot(pk_1_),
                                                         alphak_1_,
                                                         fk_ - fk_1_,
                                                         gk_.dot(pk_1_),
                                                         ls_opts_.minAlpha,
                                                         1.0));
          } else {
            // On the first step (or, after a reset) use the default step size
            alpha0_ = alpha_ = ls_opts_.alpha0;
          }

          // Perform the line search.  If successful, the results are in the
          // variables: xk_1_, fk_1_ and gk_1_.
          retCode = WolfeLineSearch(func_, alpha_, xk_1_, fk_1_, gk_1_,
                                    pk_, xk_, fk_, gk_,
                                    ls_opts_.c1, ls_opts_.c2,
                                    ls_opts_.minAlpha);
          if (retCode) {
            // Line search failed...
            if (resetB) {
              // did a Hessian reset and it still failed,
              // and nothing left to try
              retCode = TERM_LSFAIL;
              return retCode;
            } else {
              // try resetting the Hessian approximation
              resetB = 2;
              note_ += "LS failed, Hessian reset";
              continue;
            }
          } else {
            break;
          }
        }

        // Swap things so that k is the most recent iterate
        std::swap(fk_, fk_1_);
        xk_.swap(xk_1_);
        gk_.swap(gk_1_);
        pk_.swap(pk_1_);

        sk.noalias() = xk_ - xk_1_;
        yk.noalias() = gk_ - gk_1_;

        gradNorm = gk_.norm();
        stepNorm = sk.norm();

        // Update QN approximation
        if (resetB) {
          // If the QN approximation was reset, automatically scale it
          // and update the step-size accordingly
          Scalar B0fact = qn_.update(yk, sk, true);
          pk_1_ /= B0fact;
          alphak_1_ = alpha_*B0fact;
        } else {
          qn_.update(yk, sk);
          alphak_1_ = alpha_;
        }
        // Compute search direction for next step
        qn_.search_direction(pk_, gk_);

        // Check for convergence
        if (std::fabs(fk_1_ - fk_) < conv_opts_.tolAbsF) {
          // Objective function improvement wasn't sufficient
          retCode = TERM_ABSF;
        } else if (gradNorm < conv_opts_.tolAbsGrad) {
          retCode = TERM_ABSGRAD;  // Gradient norm was below threshold
        } else if (stepNorm < conv_opts_.tolAbsX) {
          retCode = TERM_ABSX;  // Change in x was too small
        } else if (itNum_ >= conv_opts_.maxIts) {
          retCode = TERM_MAXIT;  // Max number of iterations hit
        } else if (rel_obj_decrease()
                 < conv_opts_.tolRelF
                 * std::numeric_limits<Scalar>::epsilon()) {
          // Relative improvement in objective function wasn't sufficient
          retCode = TERM_RELF;
        } else if (rel_grad_norm()
                   < conv_opts_.tolRelGrad
                   * std::numeric_limits<Scalar>::epsilon()) {
          // Relative gradient norm was below threshold
          retCode = TERM_RELGRAD;
        } else {
          // Step was successful more progress to be made
          retCode = TERM_SUCCESS;
        }

        return retCode;
      }

      int minimize(VectorT &x0) {
        int retcode;
        initialize(x0);
        while (!(retcode = step()))
          continue;
        x0 = xk_;
        return retcode;
      }
    };

    template <class M>
    class ModelAdaptor {
    private:
      M& _model;
      std::vector<int> _params_i;
      std::ostream* _msgs;
      std::vector<double> _x, _g;
      size_t _fevals;

    public:
      ModelAdaptor(M& model,
                   const std::vector<int>& params_i,
                   std::ostream* msgs)
      : _model(model), _params_i(params_i), _msgs(msgs), _fevals(0) {}

      size_t fevals() const { return _fevals; }
      int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
                     double &f) {
        using Eigen::Matrix;
        using Eigen::Dynamic;
        using stan::math::index_type;
        using stan::model::log_prob_propto;
        typedef typename index_type<Matrix<double, Dynamic, 1> >::type idx_t;

        _x.resize(x.size());
        for (idx_t i = 0; i < x.size(); i++)
          _x[i] = x[i];

        try {
          f = - log_prob_propto<false>(_model, _x, _params_i, _msgs);
        } catch (const std::exception& e) {
          if (_msgs)
            (*_msgs) << e.what() << std::endl;
          return 1;
        }

        if (boost::math::isfinite(f)) {
          return 0;
        } else {
          if (_msgs)
            *_msgs << "Error evaluating model log probability: "
                      "Non-finite function evaluation." << std::endl;
          return 2;
        }
      }
      int operator()(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
                     double &f,
                     Eigen::Matrix<double, Eigen::Dynamic, 1> &g) {
        using Eigen::Matrix;
        using Eigen::Dynamic;
        using stan::math::index_type;
        using stan::model::log_prob_grad;
        typedef typename index_type<Matrix<double, Dynamic, 1> >::type idx_t;

        _x.resize(x.size());
        for (idx_t i = 0; i < x.size(); i++)
          _x[i] = x[i];

        _fevals++;

        try {
          f = - log_prob_grad<true, false>(_model, _x, _params_i, _g, _msgs);
        } catch (const std::exception& e) {
          if (_msgs)
            (*_msgs) << e.what() << std::endl;
          return 1;
        }

        g.resize(_g.size());
        for (size_t i = 0; i < _g.size(); i++) {
          if (!boost::math::isfinite(_g[i])) {
            if (_msgs)
              *_msgs << "Error evaluating model log probability: "
                                 "Non-finite gradient." << std::endl;
            return 3;
          }
          g[i] = -_g[i];
        }

        if (boost::math::isfinite(f)) {
          return 0;
        } else {
          if (_msgs)
            *_msgs << "Error evaluating model log probability: "
                   << "Non-finite function evaluation."
                   << std::endl;
          return 2;
        }
      }
      int df(const Eigen::Matrix<double, Eigen::Dynamic, 1> &x,
             Eigen::Matrix<double, Eigen:: Dynamic, 1> &g) {
        double f;
        return (*this)(x, f, g);
      }
    };

    template<typename M, typename QNUpdateType, typename Scalar = double,
             int DimAtCompile = Eigen::Dynamic>
    class BFGSLineSearch
      : public BFGSMinimizer<ModelAdaptor<M>, QNUpdateType,
                             Scalar, DimAtCompile> {
    private:
      ModelAdaptor<M> _adaptor;

    public:
      typedef BFGSMinimizer<ModelAdaptor<M>, QNUpdateType, Scalar, DimAtCompile>
      BFGSBase;
      typedef typename BFGSBase::VectorT vector_t;
      typedef typename stan::math::index_type<vector_t>::type idx_t;

      BFGSLineSearch(M& model,
                     const std::vector<double>& params_r,
                     const std::vector<int>& params_i,
                     std::ostream* msgs = 0)
        : BFGSBase(_adaptor),
          _adaptor(model, params_i, msgs) {
        initialize(params_r);
      }

      void initialize(const std::vector<double>& params_r) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> x;
        x.resize(params_r.size());
        for (size_t i = 0; i < params_r.size(); i++)
          x[i] = params_r[i];
        BFGSBase::initialize(x);
      }

      size_t grad_evals() { return _adaptor.fevals(); }
      double logp() { return -(this->curr_f()); }
      double grad_norm() { return this->curr_g().norm(); }
      void grad(std::vector<double>& g) {
        const vector_t &cg(this->curr_g());
        g.resize(cg.size());
        for (idx_t i = 0; i < cg.size(); i++)
          g[i] = -cg[i];
      }
      void params_r(std::vector<double>& x) {
        const vector_t &cx(this->curr_x());
        x.resize(cx.size());
        for (idx_t i = 0; i < cx.size(); i++)
          x[i] = cx[i];
      }
    };

  }

}

#endif
