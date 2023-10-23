data {
  int<lower=0> N; 
  int y[N];
  vector[N] x1;
}
parameters {
  real w1;
  real w2;
  real<lower=0,upper=1> robust_weight[N];
} 
model {
for (i in 1:N) {
  target+=bernoulli_logit_lpmf(y[i] | w1*x1[i] + w2)*robust_weight[i];
}
}

