data {
  int<lower=0> N;
  vector[N] kid_score;
  vector[N] work2;
  vector[N] work3;
  vector[N] work4;
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
}
model {
  beta[1] ~ uniform(70, 90);
  beta[2] ~ uniform(-10, 10);
  beta[3] ~ uniform(0, 20);
  beta[4] ~ uniform(-10, 10);
  for (i in 1:N) {
      kid_score[i] ~ normal(beta[1] + beta[2] * work2[i] + beta[3] * work3[i]
                     + beta[4] * work4[i], sigma);
  }
}
