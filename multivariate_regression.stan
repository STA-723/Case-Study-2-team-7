data {
  int<lower=0> N;  // number of observations
  int<lower=0> p;  # number of predictors
  vector[N] y;     # dependent variable
  matrix[N,p] X;   # Vector of variables
  matrix[p,p] D;   # prior for the matrix
  vector[p] beta0; # vector of priors
}

##
parameters {
  vector[p] beta;
  real<lower=0> sigma;
}

##
model {
  y ~ normal(X * beta, sigma); 
  sigma ~ cauchy(0, 0.25);
  beta ~ multi_normal(beta0, D);
}
