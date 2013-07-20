#ifndef __STAN__OPTIMIZATION__NESTEROV__GRADIENT_HPP__
#define __STAN__OPTIMIZATION__NESTEROV__GRADIENT_HPP__

#include <vector>

#include <stan/model/util.hpp>

namespace stan {

  namespace optimization {

    template <class M>
    class NesterovGradient {

    private:
      M& model_;
      std::vector<double> x_;
      std::vector<double> y_;
      std::vector<int> z_;
      double logp_;
      std::vector<double> grad_;
      double epsilon_;
      int iteration_;
      std::ostream* msgs_;

    public:
      void initialize_epsilon() {
        if (epsilon_ <= 0)
          epsilon_ = 1;
        double lastlogp = logp_;
        bool valid;
        std::vector<double> lastgrad = grad_;
        std::vector<double> lastx = x_;
        for (size_t i = 0; i < x_.size(); i++)
          x_[i] += epsilon_ * grad_[i];
        try {
          logp_ = stan::model::log_prob_grad<true,false>(model_,x_, z_, 
                                                         grad_, msgs_);
          valid = true;
        }
        catch (std::exception &ex) {
          valid = false;
        }
        if (valid && logp_ > lastlogp) {
          while (valid && logp_ > lastlogp) {
            lastlogp = logp_;
            lastgrad = grad_;
            lastx = x_;
            epsilon_ *= 2;
            for (size_t i = 0; i < x_.size(); i++)
              x_[i] += epsilon_ * grad_[i];
            try {
              logp_ = stan::model::log_prob_grad<true,false>(model_, x_, z_, 
                                                             grad_, msgs_);
            }
            catch (std::exception &ex) {
              valid = false;
            }
          }
          logp_ = lastlogp;
          grad_ = lastgrad;
          x_ = lastx;
          epsilon_ /= 2;
        } else {
          while (!valid || !(logp_ > lastlogp)) {
            epsilon_ /= 2;
            for (size_t i = 0; i < x_.size(); i++)
              x_[i] = lastx[i] + epsilon_ * lastgrad[i];
            try {
              logp_ = stan::model::log_prob_grad<true,false>(model_,x_, z_, 
                                                             grad_, msgs_);
              valid = true;
            }
            catch (std::exception &ex) {
              valid = false;
            }
          }
        }
        y_ = x_;
      }

      NesterovGradient(M& model,
                       const std::vector<double>& params_r,
                       const std::vector<int>& params_i,
                       double epsilon0 = -1,
                       std::ostream* msgs = 0) :
        model_(model), x_(params_r), y_(params_r), z_(params_i),
        epsilon_(epsilon0), iteration_(0), msgs_(msgs) {
        logp_ = stan::model::log_prob_grad<true,false>(model_,x_, z_, 
                                                       grad_, msgs_);
//  if (epsilon_ == -1)
        initialize_epsilon();
        std::cout << "epsilon = " << epsilon_ << std::endl;
      }

      double logp() { return logp_; }
      void grad(std::vector<double>& g) { g = grad_; }
      void params_r(std::vector<double>& x) { x = y_; }

      double step() {
        iteration_++;
        std::vector<double> lastx = x_;
        double lastlogp = logp_;
        bool valid = true;
        double gradnormsq = 0;
        for (size_t i = 0; i < grad_.size(); i++)
          gradnormsq += grad_[i] * grad_[i];
        epsilon_ *= 2;
        while (!valid || !(logp_ > lastlogp + 0.5 * epsilon_ * gradnormsq)) {
          epsilon_ /= 2;
          for (size_t i = 0; i < x_.size(); i++)
            x_[i] = y_[i] + epsilon_ * grad_[i];
          try {
            logp_ = model_.template log_prob<false,false>(x_, z_, msgs_);
            valid = true;
          }
          catch (std::exception &ex) {
            valid = false;
          }
        }
        for (size_t i = 0; i < x_.size(); i++)
          y_[i] = x_[i] +
            (iteration_ - 1) / (iteration_ + 2) * (x_[i] - lastx[i]);
        logp_ = stan::model::log_prob_grad<true,false>(model_,y_, z_, 
                                                       grad_, msgs_);
        return logp_;
      }
    };

  }

}

#endif
