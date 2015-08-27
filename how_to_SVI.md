HOW TO SVI in STAN
==================

1. MODIFY YOUR STAN MODEL
-------------------------

(a) INCLUDE THIS USER-DEFINED FUNCTION AT TOP of `.stan` FILE

```c++
functions {
  real divide_promote_real(int x, int y) {
    real x_real;
    x_real <- x;
    return x_real / y;
  }
}
```

(b) USE THE FOLLOWING STRUCTURE FOR THE DATA BLOCK

```c++
data {
  int<lower=0> NFULL;   // total number of datapoints in dataset
  int<lower=0> N;       // number of data points in minibatch

  int<lower=0> D;       // dimension

  vector[D] yFULL[NFULL]; // FULL dataset
  vector[D] y[N];         // minibatch
}
```


(c) COMPUTE THE MINIBATCH SCALING FACTOR

```c++
transformed data {
  real minibatch_factor;
  minibatch_factor <- divide_promote_real(N, NFULL);
}
```


(d) INCLUDE THE FACTOR AT THE END OF THE MODEL BLOCK

```c++
model{
  // prior
  // likelihood
  increment_log_prob(log(minibatch_factor));
}
```


2. COMPILE YOUR STAN MODEL INTO A .hpp FILE
-------------------------------------------

USE MY ZSHELL SCRIPT `stankeephpp`

```bash
stankeephpp () {
  inputFile=$1
  inputFileStripped=$inputFile:r
  origPWD=$PWD
  pathToProgram=$PWD/$inputFileStripped
  pathToStan=/Users/alpkucukelbir/GitHub/cmdstan/   # replace with your own path
  cd $pathToStan
  make $pathToProgram
  cd $origPWD
  rm -i -f $inputFileStripped.d
  rm -i -f $inputFileStripped.o
}
```


3. HACK THE .hpp FILE
---------------------

IMPLEMENT THE SUBSAMPLING ROUTINE

This is right after the destructor. Modify it to match whatever data structure
you have defined above in your `.stan` file.

Here's an example:
```c++
void update_minibatch() {
  // random number generator stuff for uniform sampling
  std::random_device rd;
  std::mt19937_64 gen(rd());

  // declare a uniform integer RNG
  std::uniform_int_distribution<> unif(0, NFULL - 1 );
  y.clear(); // clear whatever is in the minibatch
  int index;
  for (int n = 0; n < N; ++n) {
    index = unif(gen);
    y.push_back(yFULL.at(index)); // grab a row at random
  }
}
```


4. (RE-)COMPILE the .hpp FILE INTO AN EXECUTABLE
------------------------------------------------

RUN MY ZSHELL SCRIPT `stankeephpp` AGAIN


5. RUN with cmdstan/adsvi and stan/adsvi BRANCH
-----------------------------------------------

`./my_program variational subsample=1 data file=training_ADSVI.data.R`









