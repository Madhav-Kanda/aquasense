data {
  int<lower=0> N; 
  vector[N] y;
}
parameters {
  vector[1] beta;
  real<lower=0> sigma;
  real<lower=0, upper=1> robust_weight[N];
} 
model {
  for (i in 1:N) {
    target += normal_lpdf(y[i] | beta[1],sigma)*robust_weight[i];
  }
}
