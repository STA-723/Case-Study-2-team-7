// This script contains the Stan file to run the model with a gWishart prior
// to account for the location dependency across neighborhoods

// Introduce the data
data {
  // Number of observations N and dependent variable y
  int<lower=0> N;         
  vector[N] y;
  
  // Continuous predictors
  int<lower=0> p_cont;            // number of predictors for the continuous variables
  matrix[N,p_cont] X;             // Matrix for the continuous variables
  matrix[p_cont,p_cont] invXtX;   // Covariance matrix for the Zellner'g g prior
  vector[p_cont] beta0;           // vector of priors for the Zellner'g g prior
  
  // Discrete predictors 
  int<lower=0> p_borough;         // Dimension of the Borougs dummies
  matrix[N,p_borough] B;          // Matrix of Borough dummies
  int<lower=0> p_room;            // Dimension of the room type
  matrix[N,p_room] RType;         // Matrix of the room type
  int<lower=0> p_avl;             // Dimension of the availablity type
  matrix[N,p_avl] Avl;            // Matrix of the availablity type
  vector[N] Couple;               // Vector of Renting Couples
  
  // Neighbourhoods effect
  int<lower=0> p_neigh;           // Dimension of the Neighbourhoods dummies
  matrix[N,p_neigh] NB;           // Neighborhood matrix
  matrix[p_neigh,p_neigh] W;      // Adjacency matrix for the Neighbourhoods
  matrix[p_neigh,p_neigh] E_W;    // Diagonal matrix with adjacency counts
  vector[p_neigh] eta0;           // Prior mean of neighborhood effect
}

// Parameters
parameters {
  // Parameters on continuous variables
  vector[p_cont] beta;         // Effect of the continuous variables
  real<lower=0> tau;           // g=1/tau for the Zellner's g-prior
  
  // Parameters on Discrete variables
  vector[p_borough] beta_B;      // Effect of each Boroughs
  real mu;                       // Global mean of Boroughs
  real<lower=0> tau_B;           // g=1/tau_B for the Zellner's g-prior on Boroughs
  vector[p_room] beta_type;      // Effect of each room type
  vector[p_avl] beta_avl;        // Effect of availability type
  real beta_C;                   // Effect  Renting Couple
  
  // Global precision
  real<lower=0> phi;
  
  // Parameter on Neighborhoods 
  vector[p_neigh] eta;               // Neighborhood effect
  real<lower = -1, upper = 1> rho;   // Correlation parameter
  matrix[p_neigh,p_neigh] K;         // Spacial correlation matrix
  
}


// Model
model {
  // Likelihood
  for(n in 1:N){
    y[n] ~ normal(X[n,]*beta + B[n,]*beta_B + RType[n,]*beta_type + 
                  Avl[n,]*beta_avl + Couple[n]*beta_C + NB[n,]*eta, 1/phi);
  }
  
  // Jeffreys' prior on global precision
  target += -0.5*log(phi);
  //increment_log_prob(log(phi));
  
  // Prior on continuous variables (Zellner's g-prior)
  beta ~ multi_normal(beta0, invXtX/(tau*phi));
  tau ~ gamma(.5,.5);
  
  // Prior on Discete Variables
  for(p in 1:p_borough){
    beta_B[p] ~ normal(mu, 1/(phi*tau_B)) ; // First level of borough
  }
  mu ~ normal(0,1);                         // Global level of borough
  tau_B ~ gamma(.5, .5);                    // Zellner's g prior of boroughs
  
  for(p in 1:p_room){
    beta_type[p] ~ normal(0, 1/phi);        // Prior on room type
  }
  
  for(p in 1:p_avl){
    beta_avl[p] ~ normal(0, 1/phi);        // Prior on availability
  }  
  
  beta_C ~ normal(0, 1/phi);               // Prior on renting couples
  
  // Prior on Neighborhoods adjacency
  eta ~ multi_normal_prec(eta0, K);    // Prior on Neigh effect
  target += 0.5  * log_determinant(K) - 0.5 * sum(diagonal(inverse(E_W-rho*W) * K));  //gWishart on K
  rho ~ uniform(0,1);                      // Uniform prior on correlation 
  
}


