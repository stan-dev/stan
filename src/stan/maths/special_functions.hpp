#ifndef __STAN__MATHS__SPECIAL_FUNCTIONS_HPP__
#define __STAN__MATHS__SPECIAL_FUNCTIONS_HPP__

#include <stdexcept>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/throw_exception.hpp>

namespace stan {

  namespace maths {


    // for promote_args<> return types, see:
    // http://www.boost.org/doc/libs/1_46_0/libs/math/doc/sf_and_dist/html/math_toolkit/main_overview/result_type.html

    // C99 

    /**
     * Return the exponent base 2 of the specified argument (C99).
     *
     * The exponent base 2 function is defined by
     *
     * <code>exp2(y) = pow(2.0,y)</code>.
     *
     * @param y Value.
     * @return Exponent base 2 of value.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    exp2(T y) {
      return pow(2.0,y);
    }

    /** 
     * The positive difference function (C99).  
     *
     * The function is defined by
     *
     * <code>fdim(a,b) = (a > b) ? (a - b) : 0.0</code>.
     *
     * @param a First value.
     * @param b Second value.
     * @return Returns min(a - b, 0.0).
     */
    template <typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1, T2>::type
    fdim(T1 a, T2 b) {
      return (a > b) ? (a - b) : 0.0;
    }

    /**
     * The fused multiply-add operation (C99).  
     *
     * The function is defined by
     *
     * <code>fma(a,b,c) = (a * b) + c</code>.
     *
     * @param a First value.
     * @param b Second value.
     * @param c Third value.
     * @return Product of the first two values plust the third.
     */
    template <typename T1, typename T2, typename T3>
    inline typename boost::math::tools::promote_args<T1,T2,T3>::type
    fma(T1 a, T2 b, T3 c) {
      return a * b + c;
    }

    /**
     * Returns the base 2 logarithm of the argument (C99).
     *
     * The function is defined by:
     *
     * <code>log2(a) = log(a) / std::log(2.0)</code>.
     *
     * @param a Value.
     * @return Base 2 logarithm of the value.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log2(T a) {
      const static double LOG2 = std::log(2.0);
      return log(a) / LOG2;
    }




    // OTHER BASIC FUNCTIONS

    /**
     * The integer step, or Heaviside, function.  
     *
     * @param y Value to test.
     * @return 1 if value is greater than 0.0 and 0 otherwise.
     */
    unsigned int int_step(double y) {
      return y > 0.0;
    }

    /**
     * The step, or Heaviside, function.  
     *
     * The function is defined by 
     *
     * <code>step(y) = (y < 0.0) ? 0 : 1</code>.
     *
     * @param y Scalar argument.
     *
     * @return 1 if the specified argument is greater than or equal to
     * 0.0, and 0 otherwise.
     */
    template <typename T>
    inline int step(T y) {
      return y < 0.0 ? 0 : 1;
    }


    // PROBABILITY-RELATED FUNCTIONS
    
    /**
     * Return the log of the beta function applied to the specified
     * arguments.
     *
     * The beta function is defined for \f$a > 0\f$ and \f$b > 0\f$ by
     *
     * \f$\mbox{B}(a,b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}\f$.
     *
     * This function returns its log,
     *
     * \f$\log \mbox{B}(a,b) = \log \Gamma(a) + \log \Gamma(b) - \log \Gamma(a+b)\f$.
     *
     * See boost::math::lgamma() for the double-based and stan::agrad for the
     * variable-based log Gamma function.
     * 
     * @param a First value
     * @param b Second value
     * @return Log of the beta function applied to the two values.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    beta_log(T a, T b) {
      return lgamma(a)
	+ lgamma(b)
	- lgamma(a + b);
    }

    /**
     * Return the log of the binomial coefficient for the specified
     * arguments.
     *
     * The binomial coefficient, \f${N \choose n}\f$, read "N choose n", is
     * defined for \f$0 \leq n \leq N\f$ by
     *
     * \f${N \choose n} = \frac{N!}{n! (N-n)!}\f$.
     *
     * This function uses Gamma functions to define the log
     * and generalize the arguments to continuous N and n.
     *
     * \f$ \log {N \choose n} = \log \ \Gamma(N+1) - \log \Gamma(n+1) - \log \Gamma(N-n+1)\f$.
     *
     * @param N total number of objects.
     * @param n number of objects chosen.
     * @return log (N choose n).
     */
    template <typename T_N, typename T_n>
    inline typename boost::math::tools::promote_args<T_N, T_n>::type
    binomial_coefficient_log(T_N N, T_n n) {
      return lgamma(N + 1.0)
	- lgamma(n + 1.0)
	- lgamma(N - n + 1.0);
    }

    /**
     * Returns the inverse logit function applied to the argument.
     *
     * The inverse logit function is defined by
     *
     * \f$\mbox{logit}^{-1}(x) = \frac{1}{1 + \exp(-x)}\f$.
     *
     * This function can be used to implement the inverse link function
     * for logistic regression.
     *
     * The inverse to this function is <code>stan::maths::logit</code>.
     * 
     * @param a Argument.
     * @return Inverse logit of argument.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    inv_logit(T a) {
      return 1.0 / (1.0 + exp(-a));
    }

    /**
     * Returns the logit function applied to the
     * argument. 
     *
     * The logit function is defined as for \f$x \in [0,1]\f$ by
     * returning the log odds of \f$x\f$ treated as a probability,
     *
     * \f$\mbox{logit}(x) = \log \left( \frac{x}{1 - x} \right)\f$.
     *
     * The inverse to this function is <code>stan::maths::inv_logit</code>.
     *
     * @param a Argument.
     * @return Logit of the argument.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    logit(T a) {
      return log(a / (1.0 - a));
    }



    /**
     * The unit normal cumulative distribution function.  
     *
     * The return value for a specified input is the probability that
     * a random unit normal variate is less than or equal to the
     * specified value, defined by
     *
     * \f$\Phi(x) = \int_{-\infty}^x \mbox{\sf Norm}(x|0,1) \ dx\f$
     *
     * This function can be used to implement the inverse link function
     * for probit regression.  
     *
     * @param x Argument.
     * @return Probability random sample is less than or equal to argument. 
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    Phi(T x) {
      static const double INV_SQRT_TWO = 1.0 / std::sqrt(2.0);
      return 0.5 * (1.0 + boost::math::erf(INV_SQRT_TWO * x));
    }

    /**
     * The inverse complementary log-log function.
     *
     * The function is defined by
     *
     * <code>inv_cloglog(x) = -exp(-exp(x))</code>.
     *
     * This function can be used to implement the inverse link
     * function for complenentary-log-log regression.
     * 
     * @param x Argument.
     * @return Inverse complementary log-log of the argument.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    inv_cloglog(T x) {
      return std::exp(-std::exp(x));
    }

    /**
     * Returns the log loss function for binary classification
     * with specified reference and response values.
     *
     * The log loss function for prediction \f$\hat{y} \in [0, 1]\f$
     * given outcome \f$y \in \{ 0, 1 \}\f$ is
     *
     * \f$\mbox{logloss}(1,\hat{y}) = -\log \hat{y} \f$, and
     *
     * \f$\mbox{logloss}(0,\hat{y}) = -\log (1 - \hat{y}) \f$.
     *
     * @param y Reference value in { 0 , 1 }.
     * @param y_hat Response value in [0,1].
     * @return Log loss for response given reference value.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    binary_log_loss(int y, T y_hat) {
      return -log(y ? y_hat : (1.0 - y_hat));
    }

    // hide helper for now; could use Eigen here
    namespace {
      template <typename Vector, typename Scalar>
      int maximum(const Vector& x) {
	if(x.size() == 0)
	  BOOST_THROW_EXCEPTION(std::invalid_argument ("x must have at least one element"));
	Scalar max_x(x[0]);
	for (unsigned int i = 1; i < x.size(); ++i)
	  if (x[i] < max_x)
	    max_x = x[i];
	return max_x;
      }
    }


    /**
     * Write the values of the softmax transformed first argument
     * into the second argument.  Values in the first argument
     * are unbounded and values in the output form a simplex.
     *
     * The softmax transformed generalizes the inverse logistic
     * function by transforming a vector \f$x\f$ of length \f$K\f$ as
     *
     * \f$\mbox{softmax}(x)[i] = \frac{\exp(x[i])}{\sum_{k=1}^{K} \exp(x[k])}\f$.
     *
     * By construction, the result is a simplex, which means the values
     * are all non-negative and sum to 1.0, 
     * 
     * \f$ \sum_{k=1}^{K} \mbox{softmax}(x)[k] = 1\f$.
     *
     * The type <code>Vector</code> is for a vector with values of
     * type <code>Scalar</code>.  Vectors <code>x</code> must support
     * <code>x.size()</code> and return assignable references of type
     * <code>Scalar</code> through <code>x[int]</code>.  Variables
     * <code>a</code> of type <code>Scalar</code> need to support
     * division (in context <code>x[i] /= a</code>) and exponentiation
     * (<code>exp(a)</code>).  Conforming examples include
     *
     * <code>Vector = std::vector&lt;double&gt;</code> and
     * <code>Scalar = double</code>, 
     *
     * <code>Vector = std::vector&lt;stan::agrad::var&gt;</code> and
     * <code>Scalar = stan::agrad::var</code>, 
     * 
     * <code>Vector = Eigen::Matrix<double,Eigen::Dynamic,1>&lt;double&gt;</code> and
     * <code>Scalar = double</code>,
     *
     * and so on.
     *
     * The function <code>stan::maths::inverse_softmax</code> provides an
     * inverse of this operation up to an additive constant.  Specifically,
     * if <code>x</code> is a simplex argument, then
     *
     * <code>softmax(inv_softmax(x)) = x</code>.
     *
     * If <code>y</code> is an arbitrary vector, then we only have
     * identification up to an additive constant <code>c</code>, as in
     *
     * <code>inv_softmax(softmax(y)) = y + c * I</code>.
     *
     * @param x Input vector of unbounded parameters.
     * @param simplex Output vector of simplex values.
     * @throw std::invalid_argument if size of the input and output vectors differ.
     */
    template <typename Vector, typename Scalar>
    void softmax(const Vector& x, Vector& simplex) {
      if(x.size() != simplex.size()) 
	BOOST_THROW_EXCEPTION(std::invalid_argument ("x.size() != simplex.size()"));
      Scalar sum(0.0); 
      Scalar max_x = maximum<Vector,Scalar>(x);
      for (unsigned int i = 0; i < x.size(); ++i)
	sum += (simplex[i] = exp(x[i]-max_x));
      for (unsigned int i = 0; i < x.size(); ++i)
	simplex[i] /= sum;
    }

    
    /**
     * Writes the inverse softmax of the simplex argument into the second
     * argument.  See <code>stan::maths::softmax</code> for the inverse
     * function and a definition of the relation.
     *
     * The inverse softmax function is defined by
     *
     * \f$\mbox{inverse\_softmax}(x)[i] = \log x[i]\f$.
     *
     * This function defines the inverse of <code>stan::maths::softmax</code>
     * up to a scaling factor.
     *
     * Because of the definition, values of 0.0 in the simplex
     * are converted to negative infinity, and values of 1.0 
     * are converted to 0.0.
     *
     * There is no check that the input vector is a valid simplex vector.
     *
     * @param simplex Simplex vector input.
     * @param y Vector into which the inverse softmax is written.
     * @throw std::invalid_argument if size of the input and output vectors differ.
     */
    template <typename Vector>
    void inverse_softmax(const Vector& simplex, Vector& y) {
      if(simplex.size() != y.size())
	BOOST_THROW_EXCEPTION(std::invalid_argument ("simplex.size() != y.size()"));
      for (unsigned int i = 0; i < simplex.size(); ++i)
	y[i] = log(simplex[i]);
    }

    /**
     * Return the natural logarithm of one plus the specified value.
     *
     * The main use of this function is to cut down on intermediate
     * values during algorithmic differentiation.
     *
     * @param x Specified value.
     * @return Natural log of one plus <code>x</code>.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log1p(T x) {
      if (x < -1)
	BOOST_THROW_EXCEPTION (std::domain_error ("x can not be less than -1"));
      T absx = fabs(x);
      //double absx = fabs(stan::agrad::as_double(x));

      // Use 2nd-order Taylor approximation for very small values
      // Use 1st-order Taylor approximation for very very small values
      if (absx > 1e-9)
        return log(1 + x);
      else if (absx > 1e-16)
        return x - x*x/2;
      else
        return x;
    }

    /**
     * Return the natural logarithm of one minus the specified value.
     *
     * The main use of this function is to cut down on intermediate
     * values during algorithmic differentiation.
     *
     * @param x Specified value.
     * @return Natural log of one minus <code>x</code>.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log1m(T x) {
      return log1p(-x);
    }

    namespace {
      const double LOG_PI_OVER_FOUR = log(boost::math::constants::pi<double>()) / 4.0;
    }

    /**
     * Return the natural logarithm of the multivariate gamma function
     * with the speciifed dimensions and argument.
     *
     * <p>The multivariate gamma function \f$\Gamma_k(x)\f$ for
     * dimensionality \f$k\f$ and argument \f$x\f$ is defined by
     *
     * <p>\f$\Gamma_k(x) = \pi^{k(k-1)/4} \, \prod_{j=1}^k \Gamma(x + (1 - j)/2)\f$,
     *
     * where \f$\Gamma()\f$ is the gamma function.
     *
     * @param k Number of dimensions.
     * @param x Function argument.
     * @return Natural log of the multivariate gamma function.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    lmgamma(unsigned int k, T x) {
      typename boost::math::tools::promote_args<T>::type result 
	= k * (k - 1) * LOG_PI_OVER_FOUR;
      for (unsigned int j = 1; j <= k; ++j)
	result += lgamma(x + (1.0 - j) / 2.0);
      return result;
    }
      

    /**
     * Return the second argument if the first argument is true
     * and otherwise return the second argument.
     *
     * <p>This is just a convenience method to provide a function
     * with the same behavior as the built-in ternary operator.
     * In general, this function behaves as if defined by
     *
     * <p><code>if_else(c,y1,y0) = c ? y1 : y0</code>.
     *
     * @param c Boolean condition value.
     * @param y_true Value to return if condition is true.
     * @param y_false Value to return if condition is false.
     * @tparam B Type of conditional.
     * @tparam T Type of scalar.
     */
    template <typename B, typename T>
    T if_else(B c, T y_true, T y_false) {
      return c ? y_true : y_false;
    }

    /**
     * Return the square of the specified argument.
     *
     * <p>\f$\mbox{square}(x) = x^2\f$.
     *
     * <p>The implementation of <code>square(x)</code> is just <code>x
     * * x</code>.  Given this, this method is mainly useful in cases
     * where <code>x</code> is not a simple primitive type, particularly
     * when it is an auto-dif type.
     *
     * @param x Input to square.
     * @return Square of input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    T square(T x) {
      return x * x;
    }



  }


}

#endif
