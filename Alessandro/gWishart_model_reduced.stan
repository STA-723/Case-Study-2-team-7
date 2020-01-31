// This script contains the Stan file to run the model with a gWishart prior
// to account for the location dependency across neighborhoods

// Introduce the data
data {
  // Number of observations N and dependent variable y
  int<lower=0> N;         
  vector[N] y;
  
  // Neighbourhoods effect
  int<lower=0> p_neigh;           // Dimension of the Neighbourhoods dummies
  matrix[N,p_neigh] NB;           // Neighborhood matrix
  matrix[p_neigh,p_neigh] W;      // Adjacency matrix for the Neighbourhoods
  matrix[p_neigh,p_neigh] E_W;    // Diagonal matrix with adjacency counts
  vector[p_neigh] eta0;           // Prior mean of neighborhood effect
}

// Parameters
parameters {
  // Parameter on Neighborhoods 
  vector[p_neigh] eta;               // Neighborhood effect
  real<lower = -1, upper = 1> rho;   // Correlation parameter
  matrix[p_neigh,p_neigh] K;         // Spacial correlation matrix
  real<lower = 0> phi;
}

transformed parameters{
  matrix[p_neigh,p_neigh] D ;
  D = E_W-rho*W;
}

// Model
model {
  // Likelihood
  y ~ normal(NB*eta, 1/phi);
  // Jeffreys' prior on global precision
  //target += -0.5*log(phi);
  //increment_log_prob(log(phi));
  phi ~ gamma(.5, .5);
// Prior on Neighborhoods adjacency
  eta ~ multi_normal_prec(eta0, D);    // Prior on Neigh effect
  //target += 0.5  * log_determinant(K) - 0.5 * sum(diagonal(D * K));  //gWishart on K
  rho ~ uniform(0,1);                      // Uniform prior on correlation 
}



