data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] y_lag;
}
parameters {
  real beta[2];
  real<lower=0> sigma;
  real<lower=2, upper=10> robust_local_beta1[N];
}
model {
  for ( i in 1:N) {
      robust_local_beta1[i] ~ normal(beta[1] + beta[2] * y_lag[i], 1);
      y[i] ~ normal(robust_local_beta1[i],sigma);
  }
}
