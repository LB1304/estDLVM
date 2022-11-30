#include <RcppArmadillo.h>
#include <RcppParallel.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace RcppParallel;



// [[Rcpp::export]]
double temperature (int h, int profile, Rcpp::List profile_pars) {
  double temp;
  
  if (profile == 1) {
    double T0 = profile_pars["T0"];
    double rho = profile_pars["rho"];
    temp = T0 * pow(rho, h-1);
  } else if (profile == 2) {
    double alpha = profile_pars["alpha"];
    double beta = profile_pars["beta"];
    temp = 1 + exp(beta - h/alpha);
  } else if (profile == 3) {
    double T0 = profile_pars["T0"];
    double rho = profile_pars["rho"];
    double alpha = profile_pars["alpha"];
    double beta = profile_pars["beta"];
    double sinc = sin(3 * M_PI*M_PI/4 + h/rho)/(3 * M_PI*M_PI/4 + h/rho);
    temp = tanh(h/(2*rho)) + (T0 - beta * 2 * sqrt(2)/(3 * M_PI)) * pow(alpha, h/rho) + beta * sinc;
    if (temp < 1) {
      temp = 1;
    }
  } else {
    return 1;
  }
  
  return temp;
}



// [[Rcpp::export]]
Rcpp::List LC_em_step(arma::mat Y, arma::colvec yv, int C, int ns, int r, int n, int k, arma::rowvec piv, arma::mat Piv, arma::mat Psi, arma::cube Phi) {
  arma::rowvec ones_k(k, arma::fill::ones);
  arma::colvec ones_ns(ns, arma::fill::ones);
  
  arma::mat Num = Piv % Psi;
  arma::mat Den = arma::sum(Num, 1) * ones_k;
  arma::mat freq = yv * ones_k;
  arma::mat V = freq % (Num/Den);
  arma::rowvec b = arma::sum(V, 0);

  arma::rowvec piv_new = b/n;
  arma::mat Piv_new = ones_ns * piv_new;
  arma::cube Phi_new(C, r, k);
  for (int u = 0; u < k; u++) {
    double bu = b(u);
    for (int c = 0; c < C; c++) {
      for (int j = 0; j < r; j++) {
        arma::uvec ind = arma::find(Y.col(j) == c);
        arma::mat V_aux1 = V.col(u);
        arma::mat V_aux2 = V_aux1.rows(ind);
        double a = arma::accu(V_aux2);
        Phi_new(c, j, u) = a/bu;
      }
    }
  }
  arma::mat Psi_new(ns, k, arma::fill::ones);
  for (int j = 0; j < r; j++){
    for (int u = 0; u < k; u++) {
      arma::uvec ind = arma::conv_to<arma::uvec>::from(Y.col(j));
      arma::mat Phi_aux1 = Phi_new.slice(u);
      arma::mat Phi_aux2 = Phi_aux1.col(j);
      arma::mat Phi_aux3 = Phi_aux2.rows(ind);
      Psi_new.col(u) %= Phi_aux3;
    }
  }

  return Rcpp::List::create(Rcpp::Named("piv") = piv_new,
                            Rcpp::Named("Piv") = Piv_new,
                            Rcpp::Named("Psi") = Psi_new,
                            Rcpp::Named("Phi") = Phi_new);
}



// [[Rcpp::export]]
Rcpp::List LC_tem_step(arma::mat Y, arma::colvec yv, int C, int ns, int r, int n, int k, arma::rowvec piv, arma::mat Piv, arma::mat Psi, arma::cube Phi, double temp) {
  arma::rowvec ones_k(k, arma::fill::ones);
  arma::colvec ones_ns(ns, arma::fill::ones);
  
  arma::mat Num = pow(Piv % Psi, 1/temp);
  arma::mat Den = arma::sum(Num, 1) * ones_k;
  arma::mat freq = yv * ones_k;
  arma::mat V = freq % (Num/Den);
  arma::rowvec b = arma::sum(V, 0);
  
  arma::rowvec piv_new = b/n;
  arma::mat Piv_new = ones_ns * piv_new;
  arma::cube Phi_new(C, r, k);
  for (int u = 0; u < k; u++) {
    double bu = b(u);
    for (int c = 0; c < C; c++) {
      for (int j = 0; j < r; j++) {
        arma::uvec ind = arma::find(Y.col(j) == c);
        arma::mat V_aux1 = V.col(u);
        arma::mat V_aux2 = V_aux1.rows(ind);
        double a = arma::accu(V_aux2);
        Phi_new(c, j, u) = a/bu;
      }
    }
  }
  arma::mat Psi_new(ns, k, arma::fill::ones);
  for (int j = 0; j < r; j++){
    for (int u = 0; u < k; u++) {
      arma::uvec ind = arma::conv_to<arma::uvec>::from(Y.col(j));
      arma::mat Phi_aux1 = Phi_new.slice(u);
      arma::mat Phi_aux2 = Phi_aux1.col(j);
      arma::mat Phi_aux3 = Phi_aux2.rows(ind);
      Psi_new.col(u) %= Phi_aux3;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("piv") = piv_new,
                            Rcpp::Named("Piv") = Piv_new,
                            Rcpp::Named("Psi") = Psi_new,
                            Rcpp::Named("Phi") = Phi_new);
}



// [[Rcpp::export]]
Rcpp::List LC_em(arma::mat Y, arma::colvec yv, int k, double tol, int maxit, arma::rowvec piv, arma::mat Piv, arma::mat Psi, arma::cube Phi) {
  double lk = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
  double lko = lk - pow(10, 10);
  arma::rowvec lkv(maxit);
  int it = 0;
  
  int C = Y.max() + 1;
  int ns = Y.n_rows;
  int r = Y.n_cols;
  int n = arma::accu(yv);
  
  while (((abs(lk - lko)/abs(lko) > tol) && it < maxit) || it < 2) {
    Rcpp::List res_em_step = LC_em_step(Y, yv, C, ns, r, n,  k, piv, Piv, Psi, Phi);
    Rcpp::NumericMatrix piv_aux = res_em_step["piv"];
    piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
    Rcpp::NumericMatrix Piv_aux = res_em_step["Piv"];
    Piv = arma::mat(Piv_aux.begin(), Piv_aux.nrow(), Piv_aux.ncol(), false);
    Rcpp::NumericMatrix Psi_aux = res_em_step["Psi"];
    Psi = arma::mat(Psi_aux.begin(), Psi_aux.nrow(), Psi_aux.ncol(), false);
    Rcpp::NumericVector Phi_aux = res_em_step["Phi"];
    Phi = arma::cube(Phi_aux.begin(), C, r, k, false);
    
    lko = lk;
    lk = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    lkv(it) = lk;
    it++;
  }
  
  int np = k * r * (C - 1) + k - 1;
  double aic = -2 * lk + 2 * np;
  double bic = -2 * lk + np * log(n);
    
  return Rcpp::List::create(Rcpp::Named("lk") = lk, 
                            Rcpp::Named("lkv") = lkv.head(it), 
                            Rcpp::Named("it") = it, 
                            Rcpp::Named("piv") = piv, 
                            Rcpp::Named("Piv") = Piv, 
                            Rcpp::Named("Psi") = Psi, 
                            Rcpp::Named("Phi") = Phi, 
                            Rcpp::Named("k") = k, 
                            Rcpp::Named("np") = np, 
                            Rcpp::Named("aic") = aic, 
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List LC_tem(arma::mat Y, arma::colvec yv, int k, double tol, int maxit, arma::rowvec piv, arma::mat Piv, arma::mat Psi, arma::cube Phi, int profile, Rcpp::List profile_pars) {
  double lk = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
  double lko = lk - pow(10, 10);
  arma::rowvec lkv(maxit+1);
  int it = 0;

  int C = Y.max() + 1;
  int ns = Y.n_rows;
  int r = Y.n_cols;
  int n = arma::accu(yv);

  arma::rowvec piv_new(k);
  arma::mat Piv_new(ns, k);
  arma::mat Psi_new(ns, k, arma::fill::ones);
  arma::cube Phi_new(C, r, k);
  while (((abs(lk - lko)/abs(lko) > tol) && it < maxit) || it < 2) {
    double temp = temperature(it+1, profile, profile_pars);

    Rcpp::List res_tem_step = LC_tem_step(Y, yv, C, ns, r, n,  k, piv, Piv, Psi, Phi, temp);
    Rcpp::NumericMatrix piv_aux = res_tem_step["piv"];
    piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
    Rcpp::NumericMatrix Piv_aux = res_tem_step["Piv"];
    Piv = arma::mat(Piv_aux.begin(), Piv_aux.nrow(), Piv_aux.ncol(), false);
    Rcpp::NumericMatrix Psi_aux = res_tem_step["Psi"];
    Psi = arma::mat(Psi_aux.begin(), Psi_aux.nrow(), Psi_aux.ncol(), false);
    Rcpp::NumericVector Phi_aux = res_tem_step["Phi"];
    Phi = arma::cube(Phi_aux.begin(), C, r, k, false);
    
    lko = lk;
    lk = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    lkv(it) = lk;
    it++;
  }

  Rcpp::List res_em_step = LC_em_step(Y, yv, C, ns, r, n,  k, piv, Piv, Psi, Phi);
  Rcpp::NumericMatrix piv_aux = res_em_step["piv"];
  piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
  Rcpp::NumericMatrix Piv_aux = res_em_step["Piv"];
  Piv = arma::mat(Piv_aux.begin(), Piv_aux.nrow(), Piv_aux.ncol(), false);
  Rcpp::NumericMatrix Psi_aux = res_em_step["Psi"];
  Psi = arma::mat(Psi_aux.begin(), Psi_aux.nrow(), Psi_aux.ncol(), false);
  Rcpp::NumericVector Phi_aux = res_em_step["Phi"];
  Phi = arma::cube(Phi_aux.begin(), C, r, k, false);
  
  lko = lk;
  lk = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
  lkv(it) = lk;

  int np = k * r * (C - 1) + k - 1;
  double aic = -2 * lk + 2 * np;
  double bic = -2 * lk + np * log(n);

  return Rcpp::List::create(Rcpp::Named("lk") = lk,
                            Rcpp::Named("lkv") = lkv.head(it+1),
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Piv") = Piv,
                            Rcpp::Named("Psi") = Psi,
                            Rcpp::Named("Phi") = Phi,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("np") = np,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic,
                            Rcpp::Named("profile") = profile,
                            Rcpp::Named("profile_pars") = profile_pars);
}



// [[Rcpp::export]]
Rcpp::List Initialization_step_C (arma::mat Y, arma::colvec yv, int ns, int r, int C, int k, int n_parents) {
  Rcpp::List P(n_parents);
  arma::rowvec ones_k(k, arma::fill::ones);
  arma::colvec ones_ns(ns, arma::fill::ones);
  
  for (int b = 0; b < n_parents; b++) {
    arma::rowvec piv(k, arma::fill::randu);
    double piv_sum = arma::accu(piv);
    piv /= piv_sum;
    arma::mat Piv = ones_ns * piv;
    arma::cube Phi(C, r, k, arma::fill::randu);
    for (int u = 0; u < k; u++) {
      for (int j = 0; j < r; j++) {
        arma::mat Phi_aux1 = Phi.slice(u);
        arma::mat Phi_aux2 = Phi_aux1.col(j);
        double Phi_aux3 = arma::accu(Phi_aux2);
        Phi(arma::span::all, arma::span(j, j), arma::span(u, u)) /= Phi_aux3;
      }
    }
    arma::mat Psi(ns, k, arma::fill::ones);
    for (int u = 0; u < k; u++) {
      for (int j = 0; j < r; j++) {
        arma::uvec ind = arma::conv_to<arma::uvec>::from(Y.col(j));
        arma::mat Phi_aux1 = Phi.slice(u);
        arma::mat Phi_aux2 = Phi_aux1.col(j);
        arma::mat Phi_aux3 = Phi_aux2.rows(ind);
        Psi.col(u) %= Phi_aux3;
      }
    }

    arma::mat Num = Piv % Psi;
    arma::mat Den = arma::sum(Num, 1) * ones_k;
    arma::mat freq = yv * ones_k;
    arma::mat V = freq % (Num/Den);

    P(b) = V;
  }
  
  return P;
}



// [[Rcpp::export]]
Rcpp::List MaxExp_step_C (arma::mat Y, arma::colvec yv, int ns, int n, int r, int C, arma::mat V, int k, int R, double tol) {
  arma::rowvec ones_k(k, arma::fill::ones);
  arma::colvec ones_ns(ns, arma::fill::ones);
  
  int it = 0;
  double lk = 0;
  double lko = lk - pow(10, 10);
  
  while (((abs(lk - lko)/abs(lko) > tol) && it < R) || it < 2) {
    arma::rowvec b = arma::sum(V, 0);
    arma::rowvec piv = b/n;
    arma::mat Piv = ones_ns * piv;
    
    arma::cube Phi(C, r, k);
    for (int u = 0; u < k; u++) {
      double bu = b(u);
      for (int c = 0; c < C; c++) {
        for (int j = 0; j < r; j++) {
          arma::uvec ind = arma::find(Y.col(j) == c);
          arma::mat V_aux1 = V.col(u);
          arma::mat V_aux2 = V_aux1.rows(ind);
          double a = arma::accu(V_aux2);
          Phi(c, j, u) = a/bu;
        }
      }
    }
    
    arma::mat Psi(ns, k, arma::fill::ones);
    for (int j = 0; j < r; j++){
      for (int u = 0; u < k; u++) {
        arma::uvec ind = arma::conv_to<arma::uvec>::from(Y.col(j));
        arma::mat Phi_aux1 = Phi.slice(u);
        arma::mat Phi_aux2 = Phi_aux1.col(j);
        arma::mat Phi_aux3 = Phi_aux2.rows(ind);
        Psi.col(u) %= Phi_aux3;
      }
    }
    
    arma::mat Num = Piv % Psi;
    arma::mat Den = arma::sum(Num, 1) * ones_k;
    arma::mat freq = yv * ones_k;
    V = freq % (Num/Den);
    
    lko = lk;
    lk = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    it++;
  }
  
  return Rcpp::List::create(Rcpp::Named("V") = V, 
                            Rcpp::Named("fit") = lk);
}



// [[Rcpp::export]]
Rcpp::List LastMaxExp_step_C (arma::mat Y, arma::colvec yv, int ns, int n, int r, int C, arma::mat V, int k, int R, double tol) {
  arma::rowvec ones_k(k, arma::fill::ones);
  arma::colvec ones_ns(ns, arma::fill::ones);
  
  int it = 0;
  double lk = 0;
  double lko = lk - pow(10, 10);
  
  while (((abs(lk - lko)/abs(lko) > tol) && it < R) || it < 2) {
    arma::rowvec b = arma::sum(V, 0);
    arma::rowvec piv = b/n;
    arma::mat Piv = ones_ns * piv;
    
    arma::cube Phi(C, r, k);
    for (int u = 0; u < k; u++) {
      double bu = b(u);
      for (int c = 0; c < C; c++) {
        for (int j = 0; j < r; j++) {
          arma::uvec ind = arma::find(Y.col(j) == c);
          arma::mat V_aux1 = V.col(u);
          arma::mat V_aux2 = V_aux1.rows(ind);
          double a = arma::accu(V_aux2);
          Phi(c, j, u) = a/bu;
        }
      }
    }
    
    arma::mat Psi(ns, k, arma::fill::ones);
    for (int j = 0; j < r; j++){
      for (int u = 0; u < k; u++) {
        arma::uvec ind = arma::conv_to<arma::uvec>::from(Y.col(j));
        arma::mat Phi_aux1 = Phi.slice(u);
        arma::mat Phi_aux2 = Phi_aux1.col(j);
        arma::mat Phi_aux3 = Phi_aux2.rows(ind);
        Psi.col(u) %= Phi_aux3;
      }
    }
    
    arma::mat Num = Piv % Psi;
    arma::mat Den = arma::sum(Num, 1) * ones_k;
    arma::mat freq = yv * ones_k;
    V = freq % (Num/Den);
    
    lko = lk;
    lk = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    it++;
  }
  
  
  // Last M Step
  arma::rowvec b = arma::sum(V, 0);
  arma::rowvec piv = b/n;
  arma::mat Piv = ones_ns * piv;
  
  arma::cube Phi(C, r, k);
  for (int u = 0; u < k; u++) {
    double bu = b(u);
    for (int c = 0; c < C; c++) {
      for (int j = 0; j < r; j++) {
        arma::uvec ind = arma::find(Y.col(j) == c);
        arma::mat V_aux1 = V.col(u);
        arma::mat V_aux2 = V_aux1.rows(ind);
        double a = arma::accu(V_aux2);
        Phi(c, j, u) = a/bu;
      }
    }
  }
  
  arma::mat Psi(ns, k, arma::fill::ones);
  for (int j = 0; j < r; j++){
    for (int u = 0; u < k; u++) {
      arma::uvec ind = arma::conv_to<arma::uvec>::from(Y.col(j));
      arma::mat Phi_aux1 = Phi.slice(u);
      arma::mat Phi_aux2 = Phi_aux1.col(j);
      arma::mat Phi_aux3 = Phi_aux2.rows(ind);
      Psi.col(u) %= Phi_aux3;
    }
  }
  
  lk = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    
  int np = k * r * (C - 1) + k - 1;
  double aic = -2 * lk + 2 * np;
  double bic = -2 * lk + np * log(n);
  
  return Rcpp::List::create(Rcpp::Named("piv") = piv, 
                            Rcpp::Named("Phi") = Phi, 
                            Rcpp::Named("lk") = lk, 
                            Rcpp::Named("V") = V, 
                            Rcpp::Named("np") = np, 
                            Rcpp::Named("aic") = aic, 
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List CrossOver_step_C (int ns, Rcpp::List P, int n_children, int n_parents) {
  Rcpp::List V_children(n_children);
  
  int h = 0;
  for (int b = 0; b < n_children/2; b++) {
    int ind1 = arma::randi(arma::distr_param(0, n_parents-1));
    arma::mat V1 = P[ind1];
    int ind2 = arma::randi(arma::distr_param(0, n_parents-1));
    arma::mat V2 = P[ind2];
    
    int i = arma::randi(arma::distr_param(0, ns-1));
    
    arma::mat V_aux = V1.rows(i, ns-1);
    V1.rows(i, ns-1) = V2.rows(i, ns-1);
    V2.rows(i, ns-1) = V_aux;
    
    V_children(h) = V1;
    V_children(h+1) = V2;
    h += 2;
  }
  
  return V_children;
}



// [[Rcpp::export]]
Rcpp::List Selection_step_C (Rcpp::List PV_p, Rcpp::List PV_c, arma::rowvec fit_p, arma::rowvec fit_c, int n_parents, int n_children) {
  Rcpp::List PV(n_parents+n_children);
  arma::rowvec fit(n_parents+n_children);
  for (int i = 0; i < n_parents; i++) {
    PV[i] = PV_p[i];
    fit(i) = fit_p(i);
  }
  for (int i = 0; i < n_children; i++) {
    PV[n_parents + i] = PV_c[i];
    fit[n_parents + i] = fit_c[i];
  }
  
  arma::uvec indeces = arma::sort_index(fit, "descend");
  
  Rcpp::List PV_new(n_parents);
  arma::rowvec fit_new(n_parents);
  for (int i = 0; i < n_parents; i++) {
    int ind = indeces[i];
    PV_new[i] = PV[ind];
    fit_new(i) = fit(ind);
  }
  
  return Rcpp::List::create(Rcpp::Named("PV") = PV_new,
                            Rcpp::Named("fit") = fit_new);
}



// [[Rcpp::export]]
arma::mat Mutation_step_C (int ns, int k, arma::mat V, double prob_mut) {
  for (int i = 0; i < ns; i++) {
    double yn = arma::randu();
    if (yn < prob_mut) {
      arma::uvec indeces = arma::shuffle(arma::linspace<arma::uvec>(0, k-1, k));
      int ind1 = indeces(0);
      int ind2 = indeces(1);
      
      double aux = V(i, ind1);
      V(i, ind1) = V(i, ind2);
      V(i, ind2) = aux;
    }
  }
  
  return V;
}



// [[Rcpp::export]]
Rcpp::List LC_eem (arma::mat Y, arma::colvec yv, int k, double tol, int maxit, int n_parents, int n_children, double prob_mut, int R) {
  int ns = yv.n_rows;
  int n = arma::accu(yv);
  int r = Y.n_cols;
  int C = Y.max()+ 1;
  
  // 1. Initial values
  Rcpp::List PV1 = Initialization_step_C(Y = Y, yv = yv, ns = ns, r = r, C = C, k = k, n_parents = n_parents);
  Rcpp::List PV2(n_parents), PV3(n_children), PV4(n_children), PV5(n_parents), PV6(n_parents);
  arma::rowvec fit2(n_parents), fit4(n_children), fit5(n_parents);
  double fit_old;
  arma::mat V(ns, k);
  bool conv = false;
  int it = 0;
  
  while (!conv) {
    // 2. Update parents and compute fitness
    for (int b = 0; b < n_parents; b++) {
      Rcpp::NumericMatrix V_aux = PV1[b];
      V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
      Rcpp::List PV2_aux = MaxExp_step_C(Y, yv, ns, n, r, C, V, k, R, tol);
      PV2(b) = PV2_aux["V"];
      fit2(b) = PV2_aux["fit"];
    }
    // 3. Cross-over
    PV3 = CrossOver_step_C(ns, PV2, n_children, n_parents);
    // 4. Update children and compute fitness
    for (int b = 0; b < n_children; b++) {
      Rcpp::NumericMatrix V_aux = PV3[b];
      V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
      Rcpp::List PV4_aux = MaxExp_step_C(Y, yv, ns, n, r, C, V, k, R, tol);
      PV4(b) = PV4_aux["V"];
      fit4(b) = PV4_aux["fit"];
    }
    // 5. Select new parents
    PV5 = Selection_step_C(PV2, PV4, fit2, fit4, n_parents, n_children);
    Rcpp::NumericVector fit_aux = PV5["fit"];
    fit5 = arma::rowvec(fit_aux.begin(), fit_aux.length(), false);
    PV5 = PV5["PV"];
    // 6. Mutation
    PV6(0) = PV5(0);
    for (int b = 1; b < n_parents; b++) {
      Rcpp::NumericMatrix V_aux = PV5[b];
      V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
      PV6(b) = Mutation_step_C(ns, k, V, prob_mut);
    }

    if (it > 0) {
      conv = (abs(fit5.max() - fit_old)/abs(fit_old) < tol);
    }
    fit_old = fit5.max();
    it++;
    PV1 = PV6;
  }
  
  // 7. Last EM steps
  int ind = fit5.index_max();
  Rcpp::NumericMatrix V_aux = PV5[ind];
  V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
  Rcpp::List out = LastMaxExp_step_C(Y, yv, ns, n, r, C, V, k, maxit, tol);
      
  return out;
}







