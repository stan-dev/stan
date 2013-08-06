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
      double gamma_;
      double lambda_;
      
      std::ostream* output_stream_;

    public:
      
      void initialize_epsilon() {
        
        if (epsilon_ <= 0)
          epsilon_ = 1;
        
        bool valid = true;
        double old_logp = logp_;
        std::vector<double> old_x = x_;
        std::vector<double> old_grad = grad_;
        
        for (size_t i = 0; i < x_.size(); i++)
          x_[i] += epsilon_ * old_grad[i];
        
        try {
          logp_ = stan::model::log_prob_grad<true,false>(model_,x_, z_, 
                                                         grad_, output_stream_);
          valid = true;
        }
        catch (std::exception &ex) {
          valid = false;
        }
        
        if (valid && (logp_ > old_logp) ) {
          
          while (valid && logp_ > old_logp) {
            
            epsilon_ *= 2;
            x_ = old_x;
            
            for (size_t i = 0; i < x_.size(); i++)
              x_[i] += epsilon_ * old_grad[i];
            
            try {
              logp_ = stan::model::log_prob_grad<true,false>(model_, x_, z_, 
                                                             grad_, output_stream_);
            }
            catch (std::exception &ex) {
              valid = false;
            }
          }
          
          epsilon_ /= 2;
          
        } else {
          

          while (!valid || !(logp_ > old_logp)) {
            
            epsilon_ /= 2;
            x_ = old_x;
            
            for (size_t i = 0; i < x_.size(); i++)
              x_[i] += epsilon_ * old_grad[i];
            
            try {
              logp_ = stan::model::log_prob_grad<true,false>(model_,x_, z_, 
                                                             grad_, output_stream_);
              valid = true;
            }
            catch (std::exception &ex) {
              valid = false;
            }
          }
        }
        
        x_ = old_x;
        grad_ = old_grad;
        
      }

      NesterovGradient(M& model,
                       const std::vector<double>& params_r,
                       const std::vector<int>& params_i,
                       double epsilon = -1,
                       std::ostream* output_stream = 0) :
        model_(model),
        x_(params_r), y_(params_r), z_(params_i),
        epsilon_(epsilon), gamma_(0.0), lambda_(0.0),
        output_stream_(output_stream) {
        
        logp_ = stan::model::log_prob_grad<true,false>(model_,x_, z_,
                                                         grad_, output_stream_);
        initialize_epsilon();
          
      }

      double logp() { return logp_; }
      void grad(std::vector<double>& g) { g = grad_; }
      void params_r(std::vector<double>& x) { x = x_; }

      double step() {
        
        std::vector<double> old_y = y_;
        double old_lambda = lambda_;
        
        lambda_ = 0.5 * ( 1 + std::sqrt(1.0 + 4.0 * old_lambda * old_lambda) );
        gamma_ = (old_lambda - 1.0) / lambda_;

        for (size_t i = 0; i < x_.size(); i++)
          y_[i] = x_[i] + epsilon_ * grad_[i];
        
        for (size_t i = 0; i < x_.size(); i++)
          x_[i] = (1 - gamma_) * y_[i] + gamma_ * old_y[i];
        
        logp_ = stan::model::log_prob_grad<true,false>(model_, x_, z_,
                                                       grad_, output_stream_);
        
        return logp_;
        
      }
      
    };

  }

}

#endif
