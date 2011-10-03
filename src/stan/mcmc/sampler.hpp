#ifndef __STAN__MCMC__SAMPLER_HPP__
#define __STAN__MCMC__SAMPLER_HPP__

#include <vector>

namespace stan {

  namespace mcmc {

    namespace {
      // x' * y
      inline double dot(std::vector<double>& x,
                        std::vector<double>& y) {
	double sum = 0.0;
	for (unsigned int i = 0; i < x.size(); ++i)
	  sum += x[i] * y[i];
	return sum;
      }

      // x' * x
      inline double dot_self(std::vector<double>& x) {
	double sum = 0.0;
	for (unsigned int i = 0; i < x.size(); ++i)
	  sum += x[i] * x[i];
	return sum;
      }

      // x <- x + lambda * y
      inline void scaled_add(std::vector<double>& x, 
			     std::vector<double>& y,
			     double lambda) {
	for (unsigned int i = 0; i < x.size(); ++i)
	  x[i] += lambda * y[i];
      }

      inline double dist(const std::vector<double>& x, const std::vector<double>& y) {
	double result = 0;
	for (unsigned int i = 0; i < x.size(); ++i) {
	  double diff = x[i] - y[i];
	  result += diff * diff;
	}
	return sqrt(result);
      }

      inline double sum_vec(std::vector<double> x) {
	double sum = x[0];
	for (unsigned int i = 1; i < x.size(); ++i)
	  sum += x[i];
	return sum;
      }

      inline double max_vec(std::vector<double> x) {
	double max = x[0];
	for (unsigned int i = 1; i < x.size(); ++i)
	  if (x[i] > max)
	    max = x[i];
	return max;
      }
      
      int sample_unnorm_log(std::vector<double> probs, boost::uniform_01<boost::mt19937&>& rand_uniform_01) {
	// linearize and scale, but don't norm
	double mx = max_vec(probs);
	for (unsigned int k = 0; k < probs.size(); ++k)
	  probs[k] = exp(probs[k] - mx);

	// norm by scaling uniform sample
	double sum_probs = sum_vec(probs);
	// handles overrun due to arithmetic imprecision
	double sample_0_sum = std::max(rand_uniform_01() * sum_probs, sum_probs);  
	int k = 0;
	double cum_unnorm_prob = probs[0];
	while (cum_unnorm_prob < sample_0_sum)
	  cum_unnorm_prob += probs[++k];
	return k;
      }
    }

    /**
     * A sample consists of real parameters, integer parameters, and a
     * log probability.
     */
    class sample {
    private:
      const std::vector<double> _params_r;
      const std::vector<int> _params_i;
      const double _log_prob;

    public:

      /**
       * Construct a sample from the specified parameters and log
       * probability. 
       *
       * This constructor copies the parameter vectors so that
       * subsequent changes to the parameters do not affect the
       * sample.
       *
       * @param params_r Scalar (real) parameters.
       * @param params_i Discrete (integer) parameters.
       * @param log_prob Log probability of parameters and data in the
       * model.
       */
      sample(std::vector<double>& params_r,
	     std::vector<int>& params_i,
	     double log_prob) :
	_params_r(params_r), 
	_params_i(params_i),
	_log_prob(log_prob) {
      }
      
      /**
       * Destroy this sample.
       *
       * This base class implements is destructor as a no-op.
       */
      virtual ~sample() {
      }

      /**
       * Return the number of real (scalar) parameters.
       * 
       * @return Number of real parameters.
       */
      inline int size_r() {
	return _params_r.size();
      }

      /**
       * Return the value of the parameter at the specified
       * index.
       *
       * @param k Index of parameter.
       * @return Parameter at index.
       */
      inline double params_r(int k) {
	return _params_r[k];
      }

      /**
       * Set the specified parameter vector's value to the
       * parameters in this sample.
       * 
       * @param x Vector into which to write the parameters.
       */
      inline void params_r(std::vector<double>& x) {
	x = _params_r;
      }

      /**
       * Return the number of discrete integer (discrete) parameters.
       * 
       * @return Number of integer parameters.
       */
      inline int size_i() {
	return _params_i.size();
      }

      /**
       * Return the integer (discrete) parameter at the specified index.
       * 
       * @param k Index of parameter.
       * @return Parameter at the specified index.
       */
      inline int params_i(int k) {
	return _params_i[k];
      }

      /**
       * Set the specified parameter vector's value to
       * the parameters in this sample.
       *
       * @param n Vector into which to write parameters.
       */
      inline void params_i(std::vector<int>& n) {
	n = _params_i;
      }
  
      /**
       * Return the log probability (possibly unnormalized) for this
       * sample.
       *
       * @return Log probability.
       */
      inline double log_prob() {
	return _log_prob;
      }

    };

    /**
     * A Markov Chain Monte Carlo (MCMC) sampler abstract base class.
     */
    class sampler {
    protected:
      unsigned int _nfevals;

    public:

      /**
       * Construct a sampler.
       *
       * This is a no-op for this base class.
       */
      sampler() : _nfevals(0) { 
      }
      
      /**
       * Destroy this sampler.
       *
       * This function is implemented as a no-op for this
       * base class.
       */
      virtual ~sampler() { 
      }
      
      /**
       * Set the model real and integer parameters to the specified
       * values.  
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @param z Integer parameters.
       */
      virtual void set_params(std::vector<double> x, 
                              std::vector<int> z) = 0;

      /**
       * Return the next sample from this sampler.
       *
       * @return Next sample.
       */
      virtual sample next() = 0;

      /**
       * Return the number of times that the (possibly unnormalized)
       * log probability function has been evaluated by this sampler.
       * A useful alternative to wall time in evaluating relative
       * performance. However, it's up to the sampler implementation
       * to be sure to actually keep track of this.
       */
      unsigned int nfevals() { return _nfevals; }
    };
  }
}

#endif
