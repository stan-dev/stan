#ifndef __STAN__PROB__AUTOCORRELATION_HPP__
#define __STAN__PROB__AUTOCORRELATION_HPP__

#include <vector>
#include <complex>
#include <Eigen/FFT>

namespace stan {
  
  namespace prob {

    
    /**
     * Write autocorrelation estimates for every lag for the specified
     * input sequence into the specified result using the specified
     * FFT engine.  The return vector be resized to the same length as
     * the input sequence with lags given by array index.
     *
     * <p>The implementation involves a fast Fourier transform,
     * followed by a normalization, followed by an inverse transform.
     *
     * <p>An FFT engine can be created for reuse for type double with:
     * 
     * <blockquote><pre>
     *     Eigen::FFT<double> fft;
     * </pre></blockquote>
     *
     * @tparam T Scalar type.
     * @param y Input sequence.
     * @param ac Autocorrelations.
     * @param fft FFT engine instance.
     */
    template <typename T>
    void autocorrelation(const std::vector<T>& y,
                         std::vector<T>& ac,
                         Eigen::FFT<T>& fft) {

      using std::vector;
      using std::complex;


      size_t N = y.size();
      size_t Nt2 = 2 * N;

      // y_padded = y followed by N zeroes
      vector<T> y_padded(y);
      y_padded.insert(y_padded.end(),N,0.0);
      
      // Eigen::FFT<T> fft;
      vector<complex<T> > freqvec;
      fft.fwd(freqvec,y_padded);

      for (size_t i = 0; i < Nt2; ++i)
        freqvec[i] = complex<T>(norm(freqvec[i]), 0.0);

      fft.inv(ac,freqvec);

      ac.resize(N);
      T var = ac[0];
      for (size_t i = 0; i < N; ++i)
        ac[i] /= var;
    }

    /**
     * Write autocorrelation estimates for every lag for the specified
     * input sequence into the specified result.  The return vector be
     * resized to the same length as the input sequence with lags
     * given by array index. 
     *
     * <p>The implementation involves a fast Fourier transform,
     * followed by a normalization, followed by an inverse transform.
     *
     * <p>This method is just a light wrapper around the three-argument
     * autocorrelation function
     *
     * @tparam T Scalar type.
     * @param y Input sequence.
     * @param ac Autocorrelations.
     */
    template <typename T>
    void autocorrelation(const std::vector<T>& y,
                         std::vector<T>& ac) {
      Eigen::FFT<T> fft;
      return autocorrelation(y,ac,fft);
    }


  }
}

#endif
