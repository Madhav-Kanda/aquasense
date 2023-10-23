data {
  int<lower=0> N;
  int<lower=0,upper=1> switched[N];
  vector[N] arsenic;
  vector[N] dist100;         // rescaling
  vector[N] educ4;
}
parameters {
  vector[4] beta;
}
model {
  for (i in 1:N) {
      switched[i] ~ bernoulli_logit(beta[1] + beta[2] * dist100[i]
                             + beta[3] * arsenic[i]
                             + beta[4] * educ4[i]);
  }
}
