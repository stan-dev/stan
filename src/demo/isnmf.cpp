#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <boost/math/special_functions.hpp>
#include "stan/agrad/agrad.hpp"
#include "stan/agrad/agrad_special_functions.hpp"
#include "stan/mcmc/prob_grad_ad.hpp"
#include "stan/mcmc/hmc.hpp"
#include "stan/mcmc/adaptivehmc.hpp"
#include "stan/optimize/gradient_based.hpp"
// #include "stan/mcmc/mhmc.hpp"
// #include "stan/mcmc/shmc.hpp"
// #include "stan/mcmc/mshmc2.hpp"
// #include "stan/mcmc/mmmc.hpp"
#include "stan/mcmc/nuts.hpp"
// #include "stan/mcmc/mnuthmc.hpp"
#include "stan/mcmc/sampler.hpp"

typedef stan::agrad::var RV;

template<class T>
class Matrix {
public:
  Matrix(int M, int N)
    : M_(M), N_(N) {
    data_.assign(M*N, 0);
  }

  Matrix()
    : M_(0), N_(0) {
  }

  Matrix(Matrix<T>& X) 
    : M_(X.M()), N_(X.N()) {
    data_.assign(X.data_.begin(), X.data_.end());
  }

  void resize(int M, int N) {
    data_.assign(M*N, 0);
    M_ = M;
    N_ = N;
  }

  inline T& operator[](int i) { return data_[i]; }
  inline T& operator()(int i, int j) { return data_[i + j*M_]; }
  inline int M() { return M_; }
  inline int N() { return N_; }
  inline int size() { return M_*N_; }

  void transpose(Matrix& result) {
    result.resize(N_, M_);
    T* col = &data_[0];
    for (int n = 0; n < N_; n++) {
      T* resultrow = &result[n];
      for (int m = 0; m < M_; m++) {
        *resultrow = col[m];
        resultrow += N_;
      }
      col += M_;
    }
  }

  static void multiply(Matrix& A, Matrix& B, Matrix& result) {
    assert(A.N() == B.M());
    Matrix AT;
    A.transpose(AT);
    result.resize(A.M(), B.N());
    T* resultcol = &result[0];
    T* Bn = &B[0];
    for (int n = 0; n < B.N(); n++) {
      T* Am = &AT[0];
      for (int m = 0; m < A.M(); m++) {
        for (int i = 0; i < A.N(); i++)
          resultcol[m] += Am[i] * Bn[i];
        Am += AT.M();
      }
      Bn += B.M();
      resultcol += A.M();
    }
  }

  void print(FILE* out) {
    for (int m = 0; m < M_; m++) {
      for (int n = 0; n < N_; n++)
        fprintf(out, "%f ", (*this)(m, n));
      fprintf(out, "\n");
    }
  }

  static void load(std::ifstream& infile, Matrix& result) {
    std::vector<T> resultdata;
    int M = 0;
    int N = 0;
    std::vector<char> buffer(1024);
    while (!infile.eof()) {
      unsigned int result = 1 << 30;
      while (result >= buffer.size() - 1) {
        int lastpos = (int)infile.tellg();
        infile.getline(&buffer[0], buffer.size());
        infile.clear(infile.rdstate() & std::ios::eofbit);
        result = infile.gcount();
        if (result >= buffer.size() - 1) {
          infile.seekg(lastpos);
          buffer.resize(buffer.size()*2);
          result = buffer.size() - 1;
        }
      }
      if (result == 0)
        continue;
      char* buffptr = &buffer[0];
      while (*buffptr != 0) {
        char* nextptr;
        double scanned = strtod(buffptr, &nextptr);
        buffptr = nextptr;
        if (M == 0)
          N++;
        resultdata.push_back(scanned);
      }
      M++;
    }

    Matrix resultT(N, M);
    resultT.resize(M, N);
    for (int i = 0; i < M*N; i++)
      resultT[i] = resultdata[i];
    resultT.transpose(result);
  }

//   static void load(FILE* infile, Matrix& result) {
//     std::vector<T> resultdata;
//     int M = 0;
//     int N = 0;
//     char* buffer;
//     size_t len;
//     std::vector<char> buffer2;
//     char buffer[1000000];
//     while (!feof(infile)) {
//       buffer = fgetln(infile, &len);
//       buffer2.reserve(len);
//       memcpy(&buffer2[0], buffer, len);
//       buffer2[len-1] = 0;
//       buffer = &buffer2[0];
//       if (len == 0)
//         break;
//       char* endptr = buffer + len - 1;
// //       fprintf(stderr, "%d: buffer = %s\n", len, buffer);
//       while (buffer < endptr) {
//         double scanned = strtod(buffer, &buffer);
//         if (1) {
//           if (M == 0)
//             N++;
//           resultdata.push_back(scanned);
// //           fprintf(stderr, "%f ", resultdata.back());
//         }
//       }
//       M++;
// //       fprintf(stderr, "\n");
//     }

// //     fprintf(stderr, "M = %d, N = %d\n", M, N);
//     result.resize(M, N);
//     for (int i = 0; i < M*N; i++)
//       result[i] = resultdata[i];
//     Matrix resultT;
//     result.transpose(resultT);
//   }
  
protected:
  int M_, N_;
  std::vector<T> data_;
};

class isnmf : public stan::mcmc::prob_grad_ad {
public:
  isnmf(int K, double a, double b, Matrix<double>& X)
    : stan::mcmc::prob_grad_ad((X.M() + X.N())*K),
      X_(X), K_(K), M_(X.M()), N_(X.N()), a_(a), b_(b)
  { }

  RV log_prob(std::vector<RV>& params_r,
	      std::vector<unsigned int>& params_i) {
    Matrix<RV> W(M_, K_);
    Matrix<RV> H(K_, N_);
    for (int i = 0; i < M_*K_; i++)
      W[i] = exp(params_r[i]);
    RV* Hptr = &params_r[M_*K_];
    for (int i = 0; i < K_*N_; i++)
      H[i] = exp(Hptr[i]);

    Matrix<RV> WH;
    Matrix<RV>::multiply(W, H, WH);

    RV result = 0;
    for (int i = 0; i < W.size(); i++)
      result += a_ * log(W[i]) - a_ * W[i];
    for (int i = 0; i < H.size(); i++)
      result += b_ * log(H[i]) - b_ * H[i];
    for (int i = 0; i < WH.size(); i++)
      result += -X_[i] / WH[i] - log(WH[i]);

    return result;
  }

  double grad_log_prob(std::vector<double>& params_r,
                       std::vector<unsigned int>& params_i,
                       std::vector<double>& gradient) {
    gradient.assign(num_params_r(), 0);
    Matrix<double> W(M_, K_);
    Matrix<double> H(K_, N_);
    for (int i = 0; i < M_*K_; i++)
      W[i] = exp(params_r[i]);
    double* Hptr = &params_r[M_*K_];
    for (int i = 0; i < K_*N_; i++)
      H[i] = exp(Hptr[i]);

    Matrix<double> WH;
    Matrix<double>::multiply(W, H, WH);

    double result = 0;
    double* Wgrad = &gradient[0];
    for (int i = 0; i < W.size(); i++) {
      result += a_ * log(W[i]) - a_ * W[i];
      Wgrad[i] += a_ - a_ * W[i];
    }
    double* Hgrad = &gradient[M_*K_];
    for (int i = 0; i < H.size(); i++) {
      result += b_ * log(H[i]) - b_ * H[i];
      Hgrad[i] += b_ - b_ * H[i];
    }
    
    for (int n = 0; n < N_; n++) {
      for (int m = 0; m < M_; m++) {
        double WHmn = WH(m, n);
        double WHmninv = 1.0 / WHmn;
        double WHmninvsq = WHmninv * WHmninv;
        double Xmn = X_(m, n);
        result += -Xmn * WHmninv - log(WHmn);
        for (int k = 0; k < K_; k++) {
          double WmkHkn = W(m, k) * H(k, n);
          double gradterm = Xmn * WmkHkn * WHmninvsq - 
            WmkHkn * WHmninv;
          Wgrad[k*M_ + m] += gradterm;
          Hgrad[k + n*K_] += gradterm;
        }
      }
    }

    return result;
  }

protected:
  Matrix<double> X_;
  int K_, M_, N_;
  double a_, b_;
};

int main(int argc, char** argv) {
//   Matrix<double> A(3, 2);
//   Matrix<double> B(2, 4);
//   for (int i = 0; i < A.N(); i++)
//     for (int j = 0; j < A.M(); j++)
//       A(j, i) = i + j;
//   for (int i = 0; i < B.N(); i++)
//     for (int j = 0; j < B.M(); j++)
//       B(j, i) = i - j;
//   fprintf(stderr, "A:\n");
//   A.print(stderr);
//   fprintf(stderr, "B:\n");
//   B.print(stderr);
//   Matrix<double> AB;
//   Matrix<double>::multiply(A, B, AB);
//   fprintf(stderr, "A * B:\n");
//   AB.print(stderr);

//   FILE* infile = fopen(argv[1], "r");
//   Matrix<double> X;
//   Matrix<double>::load(infile, X);
  std::ifstream infile(argv[1], std::ifstream::in);
  Matrix<double> X;
  Matrix<double>::load(infile, X);
//   X.print(stderr);
  double a = atof(argv[2]);
  double b = atof(argv[3]);
  int K = atoi(argv[4]);

  double epsilon = 0.0025;
  int random_seed = 100003;
  int num_samples = 1000;
  int adapttime = num_samples / 2;

  isnmf model(K, a, b, X);
  std::vector<double> params_r(model.num_params_r(), 0);
  std::vector<unsigned int> params_i;
  const char* initname = "temp.dat";
  if (0) {
    srandom(random_seed);
    for (unsigned int i = 0; i < model.num_params_r(); i++)
      params_r[i] = float(random() % 1000) / 1000 - 0.5;
    //   double log_prob = stan::optimize::conjugate_gradient(params_r, params_i,
    //                                                        model, 100);
    double log_prob = model.nesterov(params_r, params_i, 0.0005, 1000);
    FILE* initfile = fopen(initname, "w");
    int nparams = params_r.size();
    fwrite(&nparams, sizeof(int), 1, initfile);
    fwrite(&params_r[0], sizeof(double), params_r.size(), initfile);
    fclose(initfile);
    return 0;
  } else {
    FILE* initfile = fopen(initname, "r");
    int numparams;
    fread(&numparams, sizeof(int), 1, initfile);
    fread(&params_r[0], sizeof(double), params_r.size(), initfile);
    fclose(initfile);
  }

  char samplertype = argv[5][0];
  stan::mcmc::sampler* sampler;
  if (samplertype == 'h') {
    unsigned int Tau = atoi(&argv[5][1]);  // number of steps
    double delta = atof(argv[6]);
    if (argc > 7)
      random_seed = atoi(argv[7]);
    sampler = new stan::mcmc::adaptivehmc(model, epsilon, Tau, delta, 
                                          adapttime, random_seed);
  } else if (samplertype == 'n') {
    double delta = atof(&argv[5][1]);
    if (argc > 6)
      random_seed = atoi(argv[6]);
    sampler = new stan::mcmc::nuts(model, epsilon, adapttime, delta,
                                   random_seed);
  } else {
    fprintf(stderr, "unrecognized sampler type %c\n", samplertype);
    return 1;
  }
  sampler->set_params(params_r, params_i);

  for (int m = 0; m < num_samples; ++m) {
    stan::mcmc::sample sample = sampler->next();
    std::vector<double> params_r;
    sample.params_r(params_r);
//     double log_prob = sample.log_prob();
    if (m % 1 == 0) {
      fprintf(stderr, "%d:  logp = %f\n", m, sample.log_prob());
//       for (unsigned int i = 0; i < model.num_params_r(); i++)
//         printf("%s%f", i == 0 ? "" : " ", params_r[i]);
//       printf("\n");
    }
  }

  fprintf(stderr, "nfevals = %d\n", sampler->nfevals());

  return 0;
}
