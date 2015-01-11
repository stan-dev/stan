#ifndef STAN__PROB__AUTOCOVARIANCE_HPP
#define STAN__PROB__AUTOCOVARIANCE_HPP

#include <stan/prob/autocorrelation.hpp>
#include <stan/math/matrix/variance.hpp>

namespace stan {
  
  namespace prob {

    
    /**
     * Write autocovariance estimates for every lag for the specified
     * input sequence into the specified result using the specified
     * FFT engine.  The return vector be resized to the same length as
     * the input sequence with lags given by array index.
     *
     * <p>The implementation involves a fast Fourier transform,
     * followed by a normalization, followed by an inverse transform.
     *
     * <p>An FFT engine can be created for reuse for type double with:
     * 
     * <pre>
     *     Eigen::FFT<double> fft;
     * </pre>
     *
     * @tparam T Scalar type.
     * @param y Input sequence.
     * @param acov Autocovariance.
     * @param fft FFT engine instance.
     */
    template <typename T>
    void autocovariance(const std::vector<T>& y,
                        std::vector<T>& acov,
                        Eigen::FFT<T>& fft) {
      
      stan::prob::autocorrelation(y, acov, fft);

      T var = stan::math::variance(y) * (y.size()-1) / y.size();
      for (size_t i = 0; i < y.size(); i++) {
        acov[i] *= var;
      }
    }

    /**
     * Write autocovariance estimates for every lag for the specified
     * input sequence into the specified result.  The return vector be
     * resized to the same length as the input sequence with lags
     * given by array index. 
     *
     * <p>The implementation involves a fast Fourier transform,
     * followed by a normalization, followed by an inverse transform.
     *
     * <p>This method is just a light wrapper around the three-argument
     * autocovariance function
     *
     * @tparam T Scalar type.
     * @param y Input sequence.
     * @param acov Autocovariances.
     */
    template <typename T>
    void autocovariance(const std::vector<T>& y,
                        std::vector<T>& acov) {
      Eigen::FFT<T> fft;
      autocovariance(y,acov,fft);
    }


  }
}

#endif
