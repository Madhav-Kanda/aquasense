data {
  int<lower=0> N; 
  vector[N] y;
}
parameters {
  vector[1] beta;
  real<lower=0> sigma;
  real<lower=20, upper=32> robust_local_beta1[N];
} 
model {
  for (i in 1:N) {
    robust_local_beta1[i] ~ normal(beta[1], 1);
    y[i] ~ normal(robust_local_beta1[1],sigma);
  }
}
