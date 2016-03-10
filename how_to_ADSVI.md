HOW TO SUBSAMPLE DATA USING ADVI
================================

0. Download an unsupported branch of Stan
-----------------------------------------

(a) Recursively clone `cmdStan` 

```
git clone --recursive git@github.com:stan-dev/cmdstan.git
```

(b) Checkout the `adsvi` branch of `cmdStan`

```
cd cmdStan
git checkout adsvi
```

(c) Checkout the `adsvi` branch of `Stan`

```
cd stan 
git checkout adsvi
```

1. Modify your Stan model
-------------------------

(a) Include this user-defined function at the top of your `.stan` file

```Stan
functions {
  real divide_promote_real(int x, int y) {
    real x_real;
    x_real <- x;
    return x_real / y;
  }
}
```

(b) Use the following variable naming structure for the data block

```Stan
data {
  int<lower=0> NFULL;     // total number of datapoints in dataset
  int<lower=0> N;         // number of data points in minibatch

  int<lower=0> D;         // dimension of each datapoint

  vector[D] yFULL[NFULL]; // FULL dataset
  vector[D] y[N];         // minibatch
}
```


(c) Compute the minibatch scaling factor

```Stan
transformed data {
  real minibatch_factor;
  minibatch_factor <- divide_promote_real(NFULL, N);
}
```


(d) Scale the log likelihood of your model

```Stan
model{
  // prior
  // likelihood
  increment_log_prob(minibatch_factor * log_likelihood_of_model);
}
```


2. Compile your Stan model into a `.hpp` file
-------------------------------------------

Here is a helpful zshell script `stanmake`

```bash
stanmake () {
  inputFile=$1
  inputFileStripped=$inputFile:r
  origPWD=$PWD
  pathToProgram=$PWD/$inputFileStripped
  pathToStan=/Users/my_username/GitHub/cmdstan/   # replace with your own path
  cd $pathToStan
  make $pathToProgram
  cd $origPWD
  rm -i -f $inputFileStripped.d
  rm -i -f $inputFileStripped.o
}
```


3. Hack the `.hpp` file
---------------------

Implement the subsampling routine in the generated `.hpp` file

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


4. (Re-)compile the `.hpp` file into an executable
------------------------------------------------

Run the zshell script `stanmake` again


5. Run 
------

`./my_program variational subsample=1 data file=training.data.R`









