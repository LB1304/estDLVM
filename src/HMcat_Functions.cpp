#include <RcppArmadillo.h>
#include <RcppParallel.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace RcppParallel;



// [[Rcpp::export(".HMcat_Temperature")]]
double HMcat_Temperature (int h, int profile, Rcpp::List profile_pars) {
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
Rcpp::List HMcat_ComputeLogLik(arma::cube S, arma::colvec yv, arma::rowvec piv, arma::cube Pi, arma::cube Phi) {
  int ns = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  int k = piv.n_cols;
  
  arma::cube Psi(ns, k, TT, arma::fill::ones);
  arma::cube L(ns, k, TT, arma::fill::zeros);
  
  for (int j = 0; j < r; j++) {
    arma::mat S_aux1 = S.slice(j);
    arma::colvec S_aux2 = S_aux1.col(0);
    arma::uvec ind = arma::conv_to<arma::uvec>::from(S_aux2);
    arma::mat Psi_aux1 = Phi.slice(j);
    arma::mat Psi_aux2 = Psi_aux1.rows(ind);
    Psi.slice(0) %= Psi_aux2;
  }
  L.slice(0) = Psi.slice(0) * arma::diagmat(piv);
  
  for(int t = 1; t < TT; t++) {
    for(int j = 0; j < r; j++) {
      arma::mat S_aux1 = S.slice(j);
      arma::colvec S_aux2 = S_aux1.col(t);
      arma::uvec ind = arma::conv_to<arma::uvec>::from(S_aux2);
      arma::mat Psi_aux1 = Phi.slice(j);
      arma::mat Psi_aux2 = Psi_aux1.rows(ind);
      Psi.slice(t) %= Psi_aux2;
      L.slice(t) = Psi.slice(t) % (L.slice(t-1) * Pi.slice(t));
    }
  }
  
  arma::colvec pv = arma::sum(L.slice(TT-1), 1);
  double LogLik = arma::accu(yv % log(pv));
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("Psi") = Psi,
                            Rcpp::Named("L") = L,
                            Rcpp::Named("pv") = pv);
}



// [[Rcpp::export]]
bool HMcat_CheckConvergence (double LogLik, double lk_old, arma::rowvec piv, arma::cube Pi, arma::cube Phi, arma::rowvec piv_old, arma::cube Pi_old, arma::cube Phi_old, int it, double tol_lk, double tol_theta, int maxit) {
  int piv_dim = piv.n_elem;
  int Pi_dim = Pi.n_elem;
  int Phi_dim = Phi.n_elem;
  int theta_dim = piv_dim + Pi_dim + Phi_dim;
  
  arma::colvec Pi_vec(Pi.memptr(), Pi_dim, false);
  arma::colvec Phi_vec(Phi.memptr(), Phi_dim, false);
  arma::colvec theta(theta_dim);
  theta.subvec(0, piv_dim-1) = arma::conv_to<arma::colvec>::from(piv);
  theta.subvec(piv_dim, piv_dim+Pi_dim-1) = Pi_vec;
  theta.subvec(piv_dim+Pi_dim, piv_dim+Pi_dim+Phi_dim-1) = Phi_vec;
  
  arma::colvec Pi_old_vec(Pi_old.memptr(), Pi_dim, false);
  arma::colvec Phi_old_vec(Phi_old.memptr(), Phi_dim, false);
  arma::colvec theta_old(theta_dim);
  theta_old.subvec(0, piv_dim-1) = arma::conv_to<arma::colvec>::from(piv_old);
  theta_old.subvec(piv_dim, piv_dim+Pi_dim-1) = Pi_old_vec;
  theta_old.subvec(piv_dim+Pi_dim, piv_dim+Pi_dim+Phi_dim-1) = Phi_old_vec;
  
  bool lk_conv = (abs(LogLik - lk_old)/abs(lk_old) < tol_lk);
  bool theta_conv = (arma::max(arma::abs(theta - theta_old)) < tol_theta);
  bool maxit_reached = (it > maxit-1);
  bool minit_done = (it > 2);
  
  bool alt = (maxit_reached + (theta_conv && lk_conv)) && minit_done;
  
  return alt;
}



// [[Rcpp::export]]
Rcpp::List HMcat_k1 (arma::cube S, arma::colvec yv, int modBasic) {
  int ns = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  int n = arma::accu(yv);
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), ns*TT, r);
  arma::rowvec Cv = arma::max(Sv, 0);
  int C = Cv.max();
  
  double piv = 1;
  double Pi = 1;
  arma::mat P(C+1, r);
  P.fill(arma::datum::nan);
  for (int j = 0; j < r; j++) {
    P(arma::span(0, Cv(j)), arma::span(j)) = arma::colvec(Cv(j)+1, arma::fill::zeros);
  }
  for (int t = 0; t < TT; t++) {
    for (int j = 0; j < r; j++) {
      for (int c = 0; c <= C; c++) {
        arma::mat S_aux = S.slice(j);
        arma::uvec ind = arma::find(S_aux.col(t) == c);
        P(c, j) += arma::accu(yv(ind));
      }
    }
  }
  arma::mat Phi = P/(n * TT);
  
  arma::cube V(ns, 1, TT, arma::fill::ones);
  
  arma::colvec pm(ns, arma::fill::ones);
  for (int t = 0; t < TT; t++) {
    for (int j = 0; j < r; j++) {
      arma::mat S_aux1 = S.slice(j);
      arma::colvec S_aux2 = S_aux1.col(t);
      arma::uvec ind = arma::conv_to<arma::uvec>::from(S_aux2);
      arma::mat Phi_aux1 = Phi.col(j);
      arma::mat Phi_aux2 = Phi_aux1.rows(ind);
      pm %= Phi_aux2;
    }
  }
  double LogLik = arma::accu(yv % log(pm));
  int N_par = r * C;
  double aic = -2 * LogLik + N_par * 2;
  double bic = -2 * LogLik + N_par * log(n);
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("LogLik_vec") = LogLik, 
                            Rcpp::Named("it") = 1,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Pi") = Pi,
                            Rcpp::Named("Phi") = Phi,
                            Rcpp::Named("k") = 1,
                            Rcpp::Named("N_par") = N_par,
                            Rcpp::Named("modBasic") = modBasic,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List HMcat_E_step(arma::colvec yv, int ns, int TT, int k, Rcpp::List llk_list, arma::cube Pi) {
  arma::cube Psi = llk_list["Psi"];
  arma::cube L = llk_list["L"];
  arma::colvec pv = llk_list["pv"];
  
  arma::cube V(ns, k, TT, arma::fill::zeros);
  arma::cube U(k, k, TT, arma::fill::zeros);
  arma::mat Yvp(ns, k);
  for (int u = 0; u < k; u++) {
    Yvp.col(u) = yv/pv;
  }
  arma::mat M(ns, k, arma::fill::ones);
  arma::mat M_new(ns, k);
  V.slice(TT-1) = Yvp % L.slice(TT-1);
  U.slice(TT-1) = (L.slice(TT-2).t() * (Yvp % Psi.slice(TT-1))) % Pi.slice(TT-1);
  if (TT > 2) {
    for (int t = TT-2; t > 0; t--) {
      M_new = (Psi.slice(t+1) % M) * Pi.slice(t+1).t();
      V.slice(t) = Yvp % L.slice(t) % M_new;
      U.slice(t) = (L.slice(t-1).t() * (Yvp % Psi.slice(t) % M_new)) % Pi.slice(t);
      M = M_new;
    }
  }
  M_new = (Psi.slice(1) % M) * Pi.slice(1).t();
  V.slice(0) = Yvp % L.slice(0) % M_new;
  
  return Rcpp::List::create(Rcpp::Named("V") = V,
                            Rcpp::Named("U") = U);
}



// [[Rcpp::export]]
Rcpp::List HMcat_TE_step(arma::colvec yv, int ns, int TT, int k, Rcpp::List llk_list, arma::cube Pi, double temp) {
  arma::cube Psi = llk_list["Psi"];
  arma::cube L = llk_list["L"];
  arma::colvec pv = llk_list["pv"];
  
  arma::cube V(ns, k, TT, arma::fill::zeros);
  arma::cube U(k, k, TT, arma::fill::zeros);
  arma::mat Yvp(ns, k);
  for (int u = 0; u < k; u++) {
    Yvp.col(u) = 1/pv;
  }
  arma::mat M(ns, k, arma::fill::ones);
  arma::mat M_new(ns, k);
  arma::mat Tmp = pow(L.slice(TT-1), 1/temp);
  arma::mat V_mat(ns, k);
  for (int u = 0; u < k; u++) {
    V_mat.col(u) = (yv/arma::sum(Tmp, 1)) % Tmp.col(u);
  }
  V.slice(TT-1) = V_mat;
  Tmp = Psi.slice(TT-1) % M;
  Tmp.each_col() /= pv;
  for (int i  = 0; i < ns; i++) {
    arma::mat L_mat = L.slice(TT-2);
    arma::colvec L_vec = arma::conv_to<arma::colvec>::from(L_mat.row(i));
    arma::mat aux = pow((L_vec * Tmp.row(i)) % Pi.slice(TT-1), 1/temp);
    U.slice(TT-1) += (yv(i)/arma::accu(aux) * aux);
  }
  if (TT > 2) {
    for (int t = TT-2; t > 0; t--) {
      M_new = (Psi.slice(t+1) % M) * Pi.slice(t+1).t();
      Tmp = pow(L.slice(t) % M_new, 1/temp);
      for (int u = 0; u < k; u++) {
        V_mat.col(u) = (yv/arma::sum(Tmp, 1)) % Tmp.col(u);
      }
      V.slice(t) = V_mat;
      Tmp = Psi.slice(t) % M_new;
      Tmp.each_col() /= pv;
      for (int i = 0; i < ns; i++) {
        arma::mat L_mat = L.slice(t-1);
        arma::colvec L_vec = arma::conv_to<arma::colvec>::from(L_mat.row(i));
        arma::mat aux = pow((L_vec * Tmp.row(i)) % Pi.slice(t), 1/temp);
        U.slice(t) += (yv(i)/arma::accu(aux) * aux);
      }
      M = M_new;
    }
  }
  M_new = (Psi.slice(1) % M) * Pi.slice(1).t();
  Tmp = pow(L.slice(0) % M_new, 1/temp);
  for (int u = 0; u < k; u++) {
    V_mat.col(u) = (yv/arma::sum(Tmp, 1)) % Tmp.col(u);
  }
  V.slice(0) = V_mat;
  
  return Rcpp::List::create(Rcpp::Named("U") = U,
                            Rcpp::Named("V") = V);
}



// [[Rcpp::export]]
Rcpp::List HMcat_M_step(arma::mat Sv, int n, int ns, int r, int TT, int k, arma::rowvec Cv, double C, Rcpp::List E_list, int modBasic) {
  arma::rowvec piv(k);
  arma::cube Pi(k, k, TT);
  arma::cube Phi(C+1, k, r);
  arma::cube V = E_list["V"];
  arma::cube U = E_list["U"];
  arma::rowvec ones_k(k, arma::fill::ones);
  
  arma::cube Y1(C+1, k, r);
  Y1.fill(arma::datum::nan);
  for (int j = 0; j < r; j++) {
    arma::colvec zeros_b(Cv(j)+1, arma::fill::zeros);
    Y1.subcube(0, 0, 0, Cv(j), 0, 0) = zeros_b;
  }
  arma::cube V_aux(ns, TT, k);
  for (int i = 0; i < ns; i++) {
    for (int u = 0; u < k; u++) {
      for (int t = 0; t < TT; t++) {
        V_aux(i, t, u) = V(i, u, t);
      }
    }
  }
  arma::mat Vv = arma::reshape(arma::mat(V_aux.memptr(), V_aux.n_elem, 1, false), ns*TT, k);
  for (int j = 0; j < r; j++) {
    for (int jb = 0; jb <= Cv(j); jb++) {
      arma::uvec ind = arma::find(Sv.col(j) == jb);
      Y1.subcube(jb, 0, j, jb, k-1, j) = arma::sum(Vv.rows(ind), 0) % ones_k;
    }
  }
  for (int j = 0; j < r; j++) {
    for (int u = 0; u < k; u++) {
      arma::colvec tmp = Y1.subcube(0, u, j, Cv(j), u, j);
      if (tmp.has_nan()) {
        arma::uvec nan_ind = arma::find_nan(tmp);
        tmp.rows(nan_ind).fill(0.0);
      }
      tmp /= arma::accu(tmp);
      arma::uvec ind = arma::find(tmp/arma::accu(tmp) <= 1e-10);
      tmp.rows(ind).fill(1e-10);
      Phi.subcube(0, u, j, Cv(j), u, j) = tmp/arma::accu(tmp);
    }
  }
  piv = arma::sum(V.slice(0), 0)/n;
  arma::uvec ind = arma::find(U <= 1e-300);
  U.elem(ind).fill(1e-300);
  if (modBasic == 0) {
    for (int t = 1; t < TT; t++) {
      Pi.slice(t) = arma::diagmat(1/arma::sum(U.slice(t), 1)) * U.slice(t);
    }
  } else if (modBasic == 1) {
    arma::mat Ut = arma::sum(U.slices(1, TT-1), 2);
    for (int t = 1; t < TT; t++) {
      Pi.slice(t) = arma::diagmat(1/arma::sum(Ut, 1)) * Ut;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("piv") = piv,
                            Rcpp::Named("Pi") = Pi,
                            Rcpp::Named("Phi") = Phi);
}



// [[Rcpp::export]]
Rcpp::List HMcat_EM(arma::cube S, arma::colvec yv, int k, double tol_lk, double tol_theta, int maxit, arma::rowvec piv, arma::cube Pi, arma::cube Phi, int modBasic) {
  Rcpp::List llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
  double LogLik = llk_list["LogLik"];
  double lk_old;
  arma::rowvec LogLik_vec(maxit);
  int it = 0;
  bool alt = false;

  int ns = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  int n = arma::accu(yv);
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), ns*TT, r);
  arma::rowvec Cv = arma::max(Sv, 0);
  int C = Cv.max();
  
  arma::rowvec piv_old(k); arma::cube Pi_old(k, k, TT); arma::cube Phi_old(C+1, k, r); arma::cube V(ns, k, TT);

  while (!alt) {
    piv_old = piv; Pi_old = Pi; Phi_old = Phi;
    Rcpp::List E_list = HMcat_E_step(yv, ns, TT, k, llk_list, Pi);
    Rcpp::List M_list = HMcat_M_step(Sv, n, ns, r, TT, k, Cv, C, E_list, modBasic);
    Rcpp::NumericMatrix piv_aux = M_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), k, false);
    Rcpp::NumericVector Pi_aux = M_list["Pi"];
    Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
    Rcpp::NumericVector Phi_aux = M_list["Phi"];
    Phi = arma::cube(Phi_aux.begin(), C+1, k, r, false);
    Rcpp::NumericVector V_aux = E_list["V"];
    V = arma::cube(V_aux.begin(), ns, k, TT, false);

    lk_old = LogLik;
    llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
    LogLik = llk_list["LogLik"];
    LogLik_vec(it) = LogLik;
    it++;
    alt = HMcat_CheckConvergence(LogLik, lk_old, piv, Pi, Phi, piv_old, Pi_old, Phi_old, it, tol_lk, tol_theta, maxit);
  }

  int N_par = (k-1) + k * arma::accu(Cv);
  if (modBasic == 0) {
    N_par += (TT - 1) * k * (k - 1);
  } else if (modBasic == 1) {
    N_par += k * (k - 1);
  } else if (modBasic > 1) {
    N_par += 2 * k * (k - 1);
  }
  double aic = -2 * LogLik + 2 * N_par;
  double bic = -2 * LogLik + N_par * log(n);

  if (arma::any(yv != 1)) {
    for (int t = 0; t < TT; t++) {
      arma::mat aux = V.slice(t);
      aux.each_col() / yv;
      V.slice(t) = aux;
    }
  }

  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("LogLik_vec") = LogLik_vec.head(it),
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Pi") = Pi,
                            Rcpp::Named("Phi") = Phi,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("N_par") = N_par,
                            Rcpp::Named("modBasic") = modBasic,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List HMcat_TEM(arma::cube S, arma::colvec yv, int k, double tol_lk, double tol_theta, int maxit, arma::rowvec piv, arma::cube Pi, arma::cube Phi, int modBasic, int profile, Rcpp::List profile_pars) {
  Rcpp::List llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
  double LogLik = llk_list["LogLik"];
  double lk_old;
  arma::rowvec LogLik_vec(maxit);
  int it = 0;
  bool alt = false;
  
  int ns = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  int n = arma::accu(yv);
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), ns*TT, r);
  arma::rowvec Cv = arma::max(Sv, 0);
  int C = Cv.max();
  
  arma::rowvec piv_old(k); arma::cube Pi_old(k, k, TT); arma::cube Phi_old(C+1, k, r); arma::cube V(ns, k, TT);
  
  while (!alt) {
    double temp = HMcat_Temperature(it+1, profile, profile_pars);
    piv_old = piv; Pi_old = Pi; Phi_old = Phi;
    Rcpp::List tE_list = HMcat_TE_step(yv, ns, TT, k, llk_list, Pi, temp);
    Rcpp::List tM_list = HMcat_M_step(Sv, n, ns, r, TT, k, Cv, C, tE_list, modBasic);
    Rcpp::NumericMatrix piv_aux = tM_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), k, false);
    Rcpp::NumericVector Pi_aux = tM_list["Pi"];
    Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
    Rcpp::NumericVector Phi_aux = tM_list["Phi"];
    Phi = arma::cube(Phi_aux.begin(), C+1, k, r, false);
    Rcpp::NumericVector V_aux = tE_list["V"];
    V = arma::cube(V_aux.begin(), ns, k, TT, false);

    lk_old = LogLik;
    llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
    LogLik = llk_list["LogLik"];
    LogLik_vec(it) = LogLik;
    it++;
    alt = HMcat_CheckConvergence(LogLik, lk_old, piv, Pi, Phi, piv_old, Pi_old, Phi_old, it, tol_lk, tol_theta, maxit-1);
  }

  Rcpp::List E_list = HMcat_E_step(yv, ns, TT, k, llk_list, Pi);
  Rcpp::List M_list = HMcat_M_step(Sv, n, ns, r, TT, k, Cv, C, E_list, modBasic);
  Rcpp::NumericMatrix piv_aux = M_list["piv"];
  piv = arma::rowvec(piv_aux.begin(), k, false);
  Rcpp::NumericVector Pi_aux = M_list["Pi"];
  Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
  Rcpp::NumericVector Phi_aux = M_list["Phi"];
  Phi = arma::cube(Phi_aux.begin(), C+1, k, r, false);
  Rcpp::NumericVector V_aux = E_list["V"];
  V = arma::cube(V_aux.begin(), ns, k, TT, false);

  lk_old = LogLik;
  llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
  LogLik = llk_list["LogLik"];
  LogLik_vec(it) = LogLik;
  it++;

  int N_par = (k-1) + k * arma::accu(Cv);
  if (modBasic == 0) {
    N_par += (TT - 1) * k * (k - 1);
  } else if (modBasic == 1) {
    N_par += k * (k - 1);
  } else if (modBasic > 1) {
    N_par += 2 * k * (k - 1);
  }
  double aic = -2 * LogLik + 2 * N_par;
  double bic = -2 * LogLik + N_par * log(n);

  if (arma::any(yv != 1)) {
    for (int t = 0; t < TT; t++) {
      arma::mat aux = V.slice(t);
      aux.each_col() / yv;
      V.slice(t) = aux;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("LogLik_vec") = LogLik_vec.head(it),
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Pi") = Pi,
                            Rcpp::Named("Phi") = Phi,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("N_par") = N_par,
                            Rcpp::Named("modBasic") = modBasic,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List HMcat_Initialization_step (arma::cube S, arma::colvec yv, int ns, int r, int TT, int k, int n_parents) {
  Rcpp::List P(n_parents);

  for (int b = 0; b < n_parents; b++) {
    arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), ns*TT, r);
    arma::rowvec Cv = arma::max(Sv, 0);
    int C = Cv.max();
    
    arma::rowvec piv(k, arma::fill::randu);
    double piv_sum = arma::accu(piv);
    piv /= piv_sum;
    arma::cube Pi(k, k, TT, arma::fill::randu);
    for (int t = 1; t < TT; t++) {
      Pi.slice(t) = arma::diagmat(1/arma::sum(Pi.slice(t), 1)) * Pi.slice(t);
    }
    Pi.slice(0) = arma::mat(k, k, arma::fill::zeros);
    arma::cube Phi(C+1, k, r);
    Phi.fill(arma::datum::nan);
    for (int j = 0; j < r; j++) {
      Phi(arma::span(0, Cv(j)), arma::span(), arma::span(j)) = arma::mat(Cv(j)+1, k, arma::fill::randu);
      for (int u = 0; u < k; u++) {
        double Phi_sum = arma::accu(Phi(arma::span(0, Cv(j)), arma::span(u), arma::span(j)));
        Phi(arma::span(0, Cv(j)), arma::span(u), arma::span(j)) /= Phi_sum;
      }
    }
    
    Rcpp::List llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
    arma::cube Psi = llk_list["Psi"];
    arma::cube L = llk_list["L"];
    arma::colvec pv = llk_list["pv"];
    
    arma::cube V(ns, k, TT, arma::fill::zeros);
    arma::cube U(k, k, TT, arma::fill::zeros);
    arma::mat Yvp(ns, k);
    for (int u = 0; u < k; u++) {
      Yvp.col(u) = yv/pv;
    }
    arma::mat M(ns, k, arma::fill::ones);
    arma::mat M_new(ns, k);
    V.slice(TT-1) = Yvp % L.slice(TT-1);
    U.slice(TT-1) = (L.slice(TT-2).t() * (Yvp % Psi.slice(TT-1))) % Pi.slice(TT-1);
    if (TT > 2) {
      for (int t = TT-2; t > 0; t--) {
        M_new = (Psi.slice(t+1) % M) * Pi.slice(t+1).t();
        V.slice(t) = Yvp % L.slice(t) % M_new;
        U.slice(t) = (L.slice(t-1).t() * (Yvp % Psi.slice(t) % M_new)) % Pi.slice(t);
        M = M_new;
      }
    }
    M_new = (Psi.slice(1) % M) * Pi.slice(1).t();
    V.slice(0) = Yvp % L.slice(0) % M_new;
    
    P(b) = Rcpp::List::create(Rcpp::Named("V") = V, Rcpp::Named("U") = U);
  }

  return P;
}



// [[Rcpp::export]]
Rcpp::List HMcat_ME_step(arma::cube S, arma::colvec yv, int k, double tol_lk, double tol_theta, int maxit, Rcpp::List P, int modBasic) {
  int ns = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  int n = arma::accu(yv);
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), ns*TT, r);
  arma::rowvec Cv = arma::max(Sv, 0);
  int C = Cv.max();
  
  arma::rowvec piv(k, arma::fill::zeros); arma::rowvec piv_old(k);
  arma::cube Pi(k, k, TT, arma::fill::zeros); arma::cube Pi_old(k, k, TT);
  arma::cube Phi(C+1, k, r, arma::fill::zeros); arma::cube Phi_old(C+1, k, r);
  double LogLik = 0; double lk_old;
  arma::cube V;
  arma::cube U;
  int it = 0;
  bool alt = false;
  
  while (!alt) {
    lk_old = LogLik; piv_old = piv; Pi_old = Pi; Phi_old = Phi;
    
    Rcpp::List M_list = HMcat_M_step(Sv, n, ns, r, TT, k, Cv, C, P, modBasic);
    Rcpp::NumericMatrix piv_aux = M_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), k, false);
    Rcpp::NumericVector Pi_aux = M_list["Pi"];
    Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
    Rcpp::NumericVector Phi_aux = M_list["Phi"];
    Phi = arma::cube(Phi_aux.begin(), C+1, k, r, false);
    Rcpp::List llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
    LogLik = llk_list["LogLik"];
    Rcpp::List E_list = HMcat_E_step(yv, ns, TT, k, llk_list, Pi);
    Rcpp::NumericVector V_aux = E_list["V"];
    V = arma::cube(V_aux.begin(), ns, k, TT, false);
    Rcpp::NumericVector U_aux = E_list["U"];
    U = arma::cube(U_aux.begin(), k, k, TT, false);
    P["V"] = V; P["U"] = U;
    
    it++;
    alt = HMcat_CheckConvergence(LogLik, lk_old, piv, Pi, Phi, piv_old, Pi_old, Phi_old, it, tol_lk, tol_theta, maxit);
  }

  return Rcpp::List::create(Rcpp::Named("V") = V,
                            Rcpp::Named("U") = U,
                            Rcpp::Named("fit") = LogLik);
}



// [[Rcpp::export]]
Rcpp::List HMcat_LastME_step (arma::cube S, arma::colvec yv, int k, double tol_lk, double tol_theta, int maxit, Rcpp::List P, int modBasic) {
  int ns = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  int n = arma::accu(yv);
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), ns*TT, r);
  arma::rowvec Cv = arma::max(Sv, 0);
  int C = Cv.max();
  
  arma::rowvec piv(k, arma::fill::zeros); arma::rowvec piv_old(k);
  arma::cube Pi(k, k, TT, arma::fill::zeros); arma::cube Pi_old(k, k, TT);
  arma::cube Phi(C+1, k, r, arma::fill::zeros); arma::cube Phi_old(C+1, k, r);
  double LogLik = 0; double lk_old;
  arma::cube V;
  arma::cube U;
  int it = 0;
  bool alt = false;
  
  while (!alt) {
    lk_old = LogLik; piv_old = piv; Pi_old = Pi; Phi_old = Phi;
    
    Rcpp::List M_list = HMcat_M_step(Sv, n, ns, r, TT, k, Cv, C, P, modBasic);
    Rcpp::NumericMatrix piv_aux = M_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), k, false);
    Rcpp::NumericVector Pi_aux = M_list["Pi"];
    Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
    Rcpp::NumericVector Phi_aux = M_list["Phi"];
    Phi = arma::cube(Phi_aux.begin(), C+1, k, r, false);
    Rcpp::List llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
    LogLik = llk_list["LogLik"];
    Rcpp::List E_list = HMcat_E_step(yv, ns, TT, k, llk_list, Pi);
    Rcpp::NumericVector V_aux = E_list["V"];
    V = arma::cube(V_aux.begin(), ns, k, TT, false);
    Rcpp::NumericVector U_aux = E_list["U"];
    U = arma::cube(U_aux.begin(), k, k, TT, false);
    P["V"] = V; P["U"] = U;
    
    it++;
    alt = HMcat_CheckConvergence(LogLik, lk_old, piv, Pi, Phi, piv_old, Pi_old, Phi_old, it, tol_lk, tol_theta, maxit);
  }
  
  
  // Last M Step
  Rcpp::List M_list = HMcat_M_step(Sv, n, ns, r, TT, k, Cv, C, P, modBasic);
  Rcpp::NumericMatrix piv_aux = M_list["piv"];
  piv = arma::rowvec(piv_aux.begin(), k, false);
  Rcpp::NumericVector Pi_aux = M_list["Pi"];
  Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
  Rcpp::NumericVector Phi_aux = M_list["Phi"];
  Phi = arma::cube(Phi_aux.begin(), C+1, k, r, false);
  Rcpp::List llk_list = HMcat_ComputeLogLik(S, yv, piv, Pi, Phi);
  LogLik = llk_list["LogLik"];
  
  
  int N_par = (k-1) + k * arma::accu(Cv);
  if (modBasic == 0) {
    N_par += (TT - 1) * k * (k - 1);
  } else if (modBasic == 1) {
    N_par += k * (k - 1);
  } else if (modBasic > 1) {
    N_par += 2 * k * (k - 1);
  }
  double aic = -2 * LogLik + 2 * N_par;
  double bic = -2 * LogLik + N_par * log(n);
  
  if (arma::any(yv != 1)) {
    for (int t = 0; t < TT; t++) {
      arma::mat aux = V.slice(t);
      aux.each_col() / yv;
      V.slice(t) = aux;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Pi") = Pi,
                            Rcpp::Named("Phi") = Phi,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("N_par") = N_par,
                            Rcpp::Named("modBasic") = modBasic,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List HMcat_CrossOver_step (int ns, int TT, Rcpp::List P, int n_children, int n_parents) {
  Rcpp::List V_children(n_children);

  int h = 0;
  for (int b = 0; b < n_children/2; b++) {
    int ind1 = arma::randi(arma::distr_param(0, n_parents-1));
    Rcpp::List gen1 = P[ind1];
    int ind2 = arma::randi(arma::distr_param(0, n_parents-1));
    Rcpp::List gen2 = P[ind2];
    arma::cube V1 = gen1["V"];
    arma::cube V2 = gen2["V"];
    arma::cube U1 = gen1["U"];
    arma::cube U2 = gen2["U"];
    
    for (int t = 0; t < TT; t++) {
      int i = arma::randi(arma::distr_param(0, ns-1));
      arma::cube V_aux = V1(arma::span(i, ns-1), arma::span(), arma::span(t));
      V1(arma::span(i, ns-1), arma::span(), arma::span(t)) = V2(arma::span(i, ns-1), arma::span(), arma::span(t));
      V2(arma::span(i, ns-1), arma::span(), arma::span(t)) = V_aux;
    }
    
    V_children(h) = Rcpp::List::create(Rcpp::Named("V") = V1, Rcpp::Named("U") = U1);
    V_children(h+1) = Rcpp::List::create(Rcpp::Named("V") = V2, Rcpp::Named("U") = U2);
    h += 2;
  }

  return V_children;
}



// [[Rcpp::export]]
Rcpp::List HMcat_Selection_step (Rcpp::List PV_p, Rcpp::List PV_c, arma::rowvec fit_p, arma::rowvec fit_c, int n_parents, int n_children) {
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
Rcpp::List HMcat_Mutation_step (int ns, int TT, int k, Rcpp::List P, double prob_mut) {
  arma::cube V = P["V"];
  arma::cube U = P["U"];
  
  for (int t = 0; t < TT; t++) {
    for (int i = 0; i < ns; i++) {
      double yn = arma::randu();
      if (yn < prob_mut) {
        arma::uvec indeces = arma::shuffle(arma::linspace<arma::uvec>(0, k-1, k));
        int ind1 = indeces(0);
        int ind2 = indeces(1);
        
        double aux = V(i, ind1, t);
        V(i, ind1, t) = V(i, ind2, t);
        V(i, ind2, t) = aux;
      }
    }
  }

  return Rcpp::List::create(Rcpp::Named("V") = V,
                            Rcpp::Named("U") = U);
}



// [[Rcpp::export]]
Rcpp::List HMcat_EEM(arma::cube S, arma::colvec yv, int k, double tol_lk, double tol_theta, int maxit, int modBasic, int n_parents, int n_children, double prob_mut, int R) {
  int ns = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;

  // 1. Initial values
  Rcpp::List PV1 = HMcat_Initialization_step(S = S, yv = yv, ns = ns, r = r, TT = TT, k = k, n_parents = n_parents);
  Rcpp::List PV2(n_parents), PV3(n_children), PV4(n_children), PV5(n_parents), PV6(n_parents);
  arma::rowvec fit2(n_parents), fit4(n_children), fit5(n_parents);
  double fit_old;
  bool conv = false;
  int it = 0;

  while (!conv) {
    // 2. Update parents and compute fitness
    for (int b = 0; b < n_parents; b++) {
      Rcpp::List P_aux = PV1[b];
      Rcpp::List PV2_aux = HMcat_ME_step(S, yv, k, tol_lk, tol_theta, R, P_aux, modBasic);
      PV2(b) = Rcpp::List::create(Rcpp::Named("V") = PV2_aux["V"], Rcpp::Named("U") = PV2_aux["U"]);
      fit2(b) = PV2_aux["fit"];
    }
    // 3. Cross-over
    PV3 = HMcat_CrossOver_step(ns, TT, PV2, n_children, n_parents);
    // 4. Update children and compute fitness
    for (int b = 0; b < n_children; b++) {
      Rcpp::List P_aux = PV3[b];
      Rcpp::List PV4_aux = HMcat_ME_step(S, yv, k, tol_lk, tol_theta, R, P_aux, modBasic);
      PV4(b) = Rcpp::List::create(Rcpp::Named("V") = PV4_aux["V"], Rcpp::Named("U") = PV4_aux["U"]);
      fit4(b) = PV4_aux["fit"];
    }
    // 5. Select new parents
    PV5 = HMcat_Selection_step(PV2, PV4, fit2, fit4, n_parents, n_children);
    Rcpp::NumericVector fit_aux = PV5["fit"];
    fit5 = arma::rowvec(fit_aux.begin(), fit_aux.length(), false);
    PV5 = PV5["PV"];
    // 6. Mutation
    PV6(0) = PV5(0);
    for (int b = 1; b < n_parents; b++) {
      Rcpp::List P_aux = PV5[b];
      PV6(b) = HMcat_Mutation_step(ns, TT, k, P_aux, prob_mut);
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
  Rcpp::List P_aux = PV5[ind];
  Rcpp::List out = HMcat_LastME_step(S, yv, k, tol_lk, tol_theta, maxit, P_aux, modBasic);

  return out;
}







