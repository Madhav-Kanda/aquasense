data {
  int<lower=0> N;
  int y[N];
  real x1[N];
}
parameters {
  real<lower=-10, upper=10> w1;
  real<lower=-10, upper=10> w2;
}
model {
  w1 ~ uniform(-10,10);
  w2 ~ uniform(-10,10);
  for (i in 1:N) {
      target += ((y[i]*(x1[i]*w1 + w2) < 0)? 0:1);
  }
}
