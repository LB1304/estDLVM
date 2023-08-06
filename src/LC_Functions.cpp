#include <RcppArmadillo.h>
#include <RcppParallel.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace RcppParallel;



// [[Rcpp::export]]
double LC_Temperature (int h, int profile, Rcpp::List profile_pars) {
  double temp;
  
  if (profile == 1) {
    double T0 = profile_pars["T0"];
    double alpha = profile_pars["alpha"];
    temp = T0 * pow(alpha, h-1);
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
bool LC_CheckConvergence (double LogLik, double LogLik_old, arma::rowvec piv, arma::cube Phi, arma::rowvec piv_old, arma::cube Phi_old, int it, double tol_lk, double tol_theta, int maxit) {
  int piv_dim = piv.n_elem;
  int Phi_dim = Phi.n_elem;
  int theta_dim = piv_dim + Phi_dim;
  
  arma::colvec Phi_vec(Phi.memptr(), Phi_dim, false);
  arma::colvec theta(theta_dim);
  theta.subvec(0, piv_dim-1) = arma::conv_to<arma::colvec>::from(piv);
  theta.subvec(piv_dim, piv_dim+Phi_dim-1) = Phi_vec;
  
  arma::colvec Phi_old_vec(Phi_old.memptr(), Phi_dim, false);
  arma::colvec theta_old(theta_dim);
  theta_old.subvec(0, piv_dim-1) = arma::conv_to<arma::colvec>::from(piv_old);
  theta_old.subvec(piv_dim, piv_dim+Phi_dim-1) = Phi_old_vec;
  
  bool LogLik_conv = (abs(LogLik - LogLik_old)/abs(LogLik_old) < tol_lk);
  bool theta_conv = (arma::max(arma::abs(theta - theta_old)) < tol_theta);
  bool maxit_reached = (it > maxit-1);
  bool minit_done = (it > 2);
  
  bool alt = (maxit_reached + (theta_conv && LogLik_conv)) && minit_done;
  
  return alt;
}



// [[Rcpp::export]]
Rcpp::List LC_k1 (arma::mat S, arma::colvec yv) {
  int C = S.max() + 1;
  int ns = S.n_rows;
  int r = S.n_cols;
  int n = arma::accu(yv);
  
  double piv = 1;
  arma::colvec Piv(ns, arma::fill::ones);
  arma::mat Phi(C, r, arma::fill::zeros);
  for (int j = 0; j < r; j++) {
    for (int c = 0; c < C; c++) {
      arma::uvec ind = arma::find(S.col(j) == c);
      Phi(c, j) = arma::accu(yv(ind))/n;
    }
  }
  arma::mat Psi(ns, 1, arma::fill::ones);
  for (int j = 0; j < r; j++){
    arma::uvec ind = arma::conv_to<arma::uvec>::from(S.col(j));
    arma::mat Phi_aux1 = Phi.col(j);
    arma::mat Phi_aux2 = Phi_aux1.rows(ind);
    Psi %= Phi_aux2;
  }
  
  double LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
  int N_par = r * (C - 1);
  double aic = -2 * LogLik + 2 * N_par;
  double bic = -2 * LogLik + N_par * log(n);
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik, 
                            Rcpp::Named("LogLik_vec") = LogLik, 
                            Rcpp::Named("it") = 1, 
                            Rcpp::Named("piv") = piv, 
                            Rcpp::Named("Phi") = Phi, 
                            Rcpp::Named("k") = 1, 
                            Rcpp::Named("N_par") = N_par, 
                            Rcpp::Named("V") = yv,
                            Rcpp::Named("aic") = aic, 
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List LC_EM_step(arma::mat S, arma::colvec yv, int C, int ns, int r, int n, int k, arma::rowvec piv, arma::mat Piv, arma::mat Psi, arma::cube Phi) {
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
        arma::uvec ind = arma::find(S.col(j) == c);
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
      arma::uvec ind = arma::conv_to<arma::uvec>::from(S.col(j));
      arma::mat Phi_aux1 = Phi_new.slice(u);
      arma::mat Phi_aux2 = Phi_aux1.col(j);
      arma::mat Phi_aux3 = Phi_aux2.rows(ind);
      Psi_new.col(u) %= Phi_aux3;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("piv") = piv_new,
                            Rcpp::Named("Piv") = Piv_new,
                            Rcpp::Named("Psi") = Psi_new,
                            Rcpp::Named("Phi") = Phi_new,
                            Rcpp::Named("V") = V);
}



// [[Rcpp::export]]
Rcpp::List LC_TEM_step(arma::mat S, arma::colvec yv, int C, int ns, int r, int n, int k, arma::rowvec piv, arma::mat Piv, arma::mat Psi, arma::cube Phi, double temp) {
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
        arma::uvec ind = arma::find(S.col(j) == c);
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
      arma::uvec ind = arma::conv_to<arma::uvec>::from(S.col(j));
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
Rcpp::List LC_EM(arma::mat S, arma::colvec yv, int k, double tol_lk, double tol_theta, int maxit, arma::rowvec piv, arma::mat Piv, arma::mat Psi, arma::cube Phi) {
  double LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
  double LogLik_old;
  arma::rowvec LogLik_vec(maxit);
  int it = 0;
  bool alt = false;
  
  int C = S.max() + 1;
  int ns = S.n_rows;
  int r = S.n_cols;
  int n = arma::accu(yv);
  
  arma::rowvec piv_old(k); arma::mat Piv_old(ns, k); arma::cube Phi_old(C, r, k); arma::mat Psi_old(ns, k); arma::mat V(ns, k);
  
  while (!alt) {
    piv_old = piv; Piv_old = Piv; Phi_old = Phi; Psi_old = Psi;
    Rcpp::List EM_list = LC_EM_step(S, yv, C, ns, r, n,  k, piv, Piv, Psi, Phi);
    Rcpp::NumericMatrix piv_aux = EM_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), k, false);
    Rcpp::NumericMatrix Piv_aux = EM_list["Piv"];
    Piv = arma::mat(Piv_aux.begin(), ns, k, false);
    Rcpp::NumericMatrix Psi_aux = EM_list["Psi"];
    Psi = arma::mat(Psi_aux.begin(), ns, k, false);
    Rcpp::NumericVector Phi_aux = EM_list["Phi"];
    Phi = arma::cube(Phi_aux.begin(), C, r, k, false);
    Rcpp::NumericMatrix V_aux = EM_list["V"];
    V = arma::mat(V_aux.begin(), ns, k, false);
    
    LogLik_old = LogLik;
    LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    LogLik_vec(it) = LogLik;
    it++;
    alt = LC_CheckConvergence(LogLik, LogLik_old, piv, Phi, piv_old, Phi_old, it, tol_lk, tol_theta, maxit);
  }
  
  int N_par = k * r * (C - 1) + k - 1;
  double aic = -2 * LogLik + 2 * N_par;
  double bic = -2 * LogLik + N_par * log(n);
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik, 
                            Rcpp::Named("LogLik_vec") = LogLik_vec.head(it), 
                            Rcpp::Named("it") = it, 
                            Rcpp::Named("piv") = piv, 
                            Rcpp::Named("Phi") = Phi, 
                            Rcpp::Named("k") = k, 
                            Rcpp::Named("N_par") = N_par, 
                            Rcpp::Named("V") = V, 
                            Rcpp::Named("aic") = aic, 
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List LC_TEM(arma::mat S, arma::colvec yv, int k, double tol_lk, double tol_theta, int maxit, arma::rowvec piv, arma::mat Piv, arma::mat Psi, arma::cube Phi, int profile, Rcpp::List profile_pars) {
  double LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
  double LogLik_old;
  arma::rowvec LogLik_vec(maxit+1);
  int it = 0;
  bool alt = false;
  
  int C = S.max() + 1;
  int ns = S.n_rows;
  int r = S.n_cols;
  int n = arma::accu(yv);
  
  arma::rowvec piv_old(k); arma::mat Piv_old(ns, k); arma::cube Phi_old(C, r, k); arma::mat Psi_old(ns, k); arma::mat V(ns, k);
  
  while (!alt) {
    double temp = LC_Temperature(it+1, profile, profile_pars);
    
    piv_old = piv; Piv_old = Piv; Phi_old = Phi; Psi_old = Psi;
    Rcpp::List res_tem_step = LC_TEM_step(S, yv, C, ns, r, n,  k, piv, Piv, Psi, Phi, temp);
    Rcpp::NumericMatrix piv_aux = res_tem_step["piv"];
    piv = arma::rowvec(piv_aux.begin(), k, false);
    Rcpp::NumericMatrix Piv_aux = res_tem_step["Piv"];
    Piv = arma::mat(Piv_aux.begin(), ns, k, false);
    Rcpp::NumericMatrix Psi_aux = res_tem_step["Psi"];
    Psi = arma::mat(Psi_aux.begin(), ns, k, false);
    Rcpp::NumericVector Phi_aux = res_tem_step["Phi"];
    Phi = arma::cube(Phi_aux.begin(), C, r, k, false);
    
    LogLik_old = LogLik;
    LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    LogLik_vec(it) = LogLik;
    it++;
    alt = LC_CheckConvergence(LogLik, LogLik_old, piv, Phi, piv_old, Phi_old, it, tol_lk, tol_theta, maxit-1);
  }
  
  Rcpp::List res_em_step = LC_EM_step(S, yv, C, ns, r, n,  k, piv, Piv, Psi, Phi);
  Rcpp::NumericMatrix piv_aux = res_em_step["piv"];
  piv = arma::rowvec(piv_aux.begin(), k, false);
  Rcpp::NumericMatrix Piv_aux = res_em_step["Piv"];
  Piv = arma::mat(Piv_aux.begin(), ns, k, false);
  Rcpp::NumericMatrix Psi_aux = res_em_step["Psi"];
  Psi = arma::mat(Psi_aux.begin(), ns, k, false);
  Rcpp::NumericVector Phi_aux = res_em_step["Phi"];
  Phi = arma::cube(Phi_aux.begin(), C, r, k, false);
  
  LogLik_old = LogLik;
  LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
  LogLik_vec(it) = LogLik;
  it++;
  
  int N_par = k * r * (C - 1) + k - 1;
  double aic = -2 * LogLik + 2 * N_par;
  double bic = -2 * LogLik + N_par * log(n);
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("LogLik_vec") = LogLik_vec.head(it),
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Phi") = Phi,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("N_par") = N_par,
                            Rcpp::Named("V") = V, 
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List LC_Initialization_step (arma::mat S, arma::colvec yv, int ns, int r, int C, int k, int n_parents) {
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
        arma::uvec ind = arma::conv_to<arma::uvec>::from(S.col(j));
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
Rcpp::List LC_ME_step (arma::mat S, arma::colvec yv, int ns, int n, int r, int C, arma::mat V, int k, int maxit, double tol_lk, double tol_theta) {
  arma::rowvec ones_k(k, arma::fill::ones);
  arma::colvec ones_ns(ns, arma::fill::ones);
  
  arma::rowvec piv(k, arma::fill::zeros); arma::rowvec piv_old(k);
  arma::mat Piv(ns, k, arma::fill::zeros); arma::mat Piv_old(ns, k);
  arma::cube Phi(C, r, k, arma::fill::zeros); arma::cube Phi_old(C, r, k);
  arma::mat Psi(ns, k, arma::fill::zeros); arma::mat Psi_old(ns, k);
  double LogLik = 0; double LogLik_old;
  int it = 0;
  bool alt = false;
  
  while (!alt) {
    LogLik_old = LogLik; piv_old = piv; Piv_old = Piv; Phi_old = Phi; Psi_old = Psi;
    
    arma::rowvec b = arma::sum(V, 0);
    piv = b/n;
    Piv = ones_ns * piv;
    
    for (int u = 0; u < k; u++) {
      double bu = b(u);
      for (int c = 0; c < C; c++) {
        for (int j = 0; j < r; j++) {
          arma::uvec ind = arma::find(S.col(j) == c);
          arma::mat V_aux1 = V.col(u);
          arma::mat V_aux2 = V_aux1.rows(ind);
          double a = arma::accu(V_aux2);
          Phi(c, j, u) = a/bu;
        }
      }
    }
    
    Psi.ones();
    for (int j = 0; j < r; j++){
      for (int u = 0; u < k; u++) {
        arma::uvec ind = arma::conv_to<arma::uvec>::from(S.col(j));
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
    
    LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    it++;
    alt = LC_CheckConvergence(LogLik, LogLik_old, piv, Phi, piv_old, Phi_old, it, tol_lk, tol_theta, maxit);
  }
  
  return Rcpp::List::create(Rcpp::Named("V") = V, 
                            Rcpp::Named("fit") = LogLik);
}



// [[Rcpp::export]]
Rcpp::List LC_LastME_step (arma::mat S, arma::colvec yv, int ns, int n, int r, int C, arma::mat V, int k, int maxit, double tol_lk, double tol_theta) {
  arma::rowvec ones_k(k, arma::fill::ones);
  arma::colvec ones_ns(ns, arma::fill::ones);
  
  arma::rowvec piv(k, arma::fill::zeros); arma::rowvec piv_old(k);
  arma::mat Piv(ns, k, arma::fill::zeros); arma::mat Piv_old(ns, k);
  arma::cube Phi(C, r, k, arma::fill::zeros); arma::cube Phi_old(C, r, k);
  arma::mat Psi(ns, k, arma::fill::zeros); arma::mat Psi_old(ns, k);
  double LogLik = 0; double LogLik_old;
  int it = 0;
  bool alt = false;
  
  while (!alt) {
    LogLik_old = LogLik; piv_old = piv; Piv_old = Piv; Phi_old = Phi; Psi_old = Psi;
    
    arma::rowvec b = arma::sum(V, 0);
    arma::rowvec piv = b/n;
    arma::mat Piv = ones_ns * piv;
    
    arma::cube Phi(C, r, k);
    for (int u = 0; u < k; u++) {
      double bu = b(u);
      for (int c = 0; c < C; c++) {
        for (int j = 0; j < r; j++) {
          arma::uvec ind = arma::find(S.col(j) == c);
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
        arma::uvec ind = arma::conv_to<arma::uvec>::from(S.col(j));
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
    
    LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
    it++;
    alt = LC_CheckConvergence(LogLik, LogLik_old, piv, Phi, piv_old, Phi_old, it, tol_lk, tol_theta, maxit);
  }
  
  
  // Last M Step
  arma::rowvec b = arma::sum(V, 0);
  piv = b/n;
  Piv = ones_ns * piv;
  
  for (int u = 0; u < k; u++) {
    double bu = b(u);
    for (int c = 0; c < C; c++) {
      for (int j = 0; j < r; j++) {
        arma::uvec ind = arma::find(S.col(j) == c);
        arma::mat V_aux1 = V.col(u);
        arma::mat V_aux2 = V_aux1.rows(ind);
        double a = arma::accu(V_aux2);
        Phi(c, j, u) = a/bu;
      }
    }
  }
  
  Psi.ones();
  for (int j = 0; j < r; j++){
    for (int u = 0; u < k; u++) {
      arma::uvec ind = arma::conv_to<arma::uvec>::from(S.col(j));
      arma::mat Phi_aux1 = Phi.slice(u);
      arma::mat Phi_aux2 = Phi_aux1.col(j);
      arma::mat Phi_aux3 = Phi_aux2.rows(ind);
      Psi.col(u) %= Phi_aux3;
    }
  }
  
  LogLik = arma::accu(yv % log(arma::sum(Psi % Piv, 1)));
  
  int N_par = k * r * (C - 1) + k - 1;
  double aic = -2 * LogLik + 2 * N_par;
  double bic = -2 * LogLik + N_par * log(n);
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik, 
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Phi") = Phi, 
                            Rcpp::Named("k") = k, 
                            Rcpp::Named("N_par") = N_par, 
                            Rcpp::Named("V") = V, 
                            Rcpp::Named("aic") = aic, 
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List LC_CrossOver_step (int ns, Rcpp::List P, int n_children, int n_parents) {
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
Rcpp::List LC_Selection_step (Rcpp::List PV_p, Rcpp::List PV_c, arma::rowvec fit_p, arma::rowvec fit_c, int n_parents, int n_children) {
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
  
  // Da sistemare riga seguente
  arma::uvec indeces = arma::sort_index(fit.replace(arma::datum::nan, R_NegInf), "descend");
  
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
arma::mat LC_Mutation_step (int ns, int k, arma::mat V, double prob_mut) {
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
Rcpp::List LC_EEM (arma::mat S, arma::colvec yv, int k, double tol_lk, double tol_theta, int maxit, int n_parents, int n_children, double prob_mut, int R) {
  int ns = yv.n_rows;
  int n = arma::accu(yv);
  int r = S.n_cols;
  int C = S.max() + 1;
  
  // 1. Initial values
  Rcpp::List PV1 = LC_Initialization_step(S = S, yv = yv, ns = ns, r = r, C = C, k = k, n_parents = n_parents);
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
      Rcpp::List PV2_aux = LC_ME_step(S, yv, ns, n, r, C, V, k, R, tol_lk, tol_theta);
      PV2(b) = PV2_aux["V"];
      fit2(b) = PV2_aux["fit"];
    }
    // 3. Cross-over
    PV3 = LC_CrossOver_step(ns, PV2, n_children, n_parents);
    // 4. Update children and compute fitness
    for (int b = 0; b < n_children; b++) {
      Rcpp::NumericMatrix V_aux = PV3[b];
      V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
      Rcpp::List PV4_aux = LC_ME_step(S, yv, ns, n, r, C, V, k, R, tol_lk, tol_theta);
      PV4(b) = PV4_aux["V"];
      fit4(b) = PV4_aux["fit"];
    }
    // 5. Select new parents
    PV5 = LC_Selection_step(PV2, PV4, fit2, fit4, n_parents, n_children);
    Rcpp::NumericVector fit_aux = PV5["fit"];
    fit5 = arma::rowvec(fit_aux.begin(), fit_aux.length(), false);
    PV5 = PV5["PV"];
    // 6. Mutation
    PV6(0) = PV5(0);
    for (int b = 1; b < n_parents; b++) {
      Rcpp::NumericMatrix V_aux = PV5[b];
      V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
      PV6(b) = LC_Mutation_step(ns, k, V, prob_mut);
    }
    
    if (it > 0) {
      conv = (abs(fit5.max() - fit_old)/abs(fit_old) < tol_lk);
    }
    fit_old = fit5.max();
    it++;
    PV1 = PV6;
  }
  
  // 7. Last EM steps
  int ind = fit5.index_max();
  Rcpp::NumericMatrix V_aux = PV5[ind];
  V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
  Rcpp::List out = LC_LastME_step(S, yv, ns, n, r, C, V, k, maxit, tol_lk, tol_theta);
  
  return out;
}
