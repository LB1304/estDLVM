#include <RcppArmadillo.h>
#include <RcppParallel.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace RcppParallel;



// [[Rcpp::export]]
double HMcont_Temperature (int h, int profile, Rcpp::List profile_pars) {
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
arma::colvec dmvnorm (arma::mat x, arma::colvec mean, arma::mat sigma) {
  int p = x.n_cols;
  
  arma::mat dec = arma::chol(sigma);
  
  arma::mat A = dec.t();
  arma::mat B = x.t();
  B.each_col() -= mean;
  
  arma::mat tmp = arma::solve(arma::trimatl(A), B);
  arma::colvec rss = arma::sum(arma::pow(tmp.t(), 2), 1);
  
  arma::colvec logretval = -arma::accu(log(arma::diagvec(dec))) - 0.5 * p * log(2 * M_PI) - 0.5 * rss;
  
  return exp(logretval);
}



// [[Rcpp::export]]
Rcpp::List HMcont_ComputeLogLik(arma::cube S, arma::rowvec piv, arma::cube Pi, arma::mat Mu, arma::mat Si) {
  int n = S.n_rows;
  int TT = S.n_cols;
  int k = piv.n_cols;
  
  arma::cube Psi(n, k, TT, arma::fill::ones);
  arma::cube L(n, k, TT, arma::fill::zeros);
  arma::colvec Lt(n);
  double LogLik;
  
  for (int u = 0; u < k; u++) {
    for (int t = 0; t < TT; t++) {
      arma::mat Yt = S(arma::span(), arma::span(t), arma::span());
      arma::colvec aux = dmvnorm(Yt, Mu.col(u), Si);
      Psi(arma::span(), arma::span(u), arma::span(t)) = aux;
    }
  }
  
  L.slice(0) = Psi.slice(0) * arma::diagmat(piv);
  Lt = arma::sum(L.slice(0), 1);
  LogLik = arma::accu(log(Lt));
  for (int u = 0; u < k; u++) {
    L(arma::span(), arma::span(u), arma::span(0)) /= Lt;
  }
  
  for(int t = 1; t < TT; t++) {
    L.slice(t) = Psi.slice(t) % (L.slice(t-1) * Pi.slice(t));
    Lt = arma::sum(L.slice(t), 1);
    LogLik += arma::accu(log(Lt));
    for (int u = 0; u < k; u++) {
      L(arma::span(), arma::span(u), arma::span(t)) /= Lt;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("Psi") = Psi,
                            Rcpp::Named("L") = L);
}



// [[Rcpp::export]]
bool HMcont_CheckConvergence (double lk, double lk_old, arma::rowvec piv, arma::cube Pi, arma::mat Mu, arma::mat Si, arma::rowvec piv_old, arma::cube Pi_old, arma::mat Mu_old, arma::mat Si_old, int it, double tol_lk, double tol_theta, int maxit) {
  int piv_dim = piv.n_elem;
  int Pi_dim = Pi.n_elem;
  //int Mu_dim = Mu.n_elem;
  //int Si_dim = Si.n_elem;
  int theta_dim = piv_dim + Pi_dim;
  //int theta_dim = piv_dim + Pi_dim + Mu_dim + Si_dim;

  arma::colvec Pi_vec(Pi.memptr(), Pi_dim, false);
  //arma::colvec Mu_vec(Mu.memptr(), Mu_dim, false);
  //arma::colvec Si_vec(Si.memptr(), Si_dim, false);
  arma::colvec theta(theta_dim);
  theta.subvec(0, piv_dim-1) = arma::conv_to<arma::colvec>::from(piv);
  theta.subvec(piv_dim, piv_dim+Pi_dim-1) = Pi_vec;
  //theta.subvec(piv_dim+Pi_dim, piv_dim+Pi_dim+Mu_dim-1) = Mu_vec;
  //theta.subvec(piv_dim+Pi_dim+Mu_dim, piv_dim+Pi_dim+Mu_dim+Si_dim-1) = Si_vec;

  arma::colvec Pi_old_vec(Pi_old.memptr(), Pi_dim, false);
  //arma::colvec Mu_old_vec(Mu_old.memptr(), Mu_dim, false);
  //arma::colvec Si_old_vec(Si_old.memptr(), Si_dim, false);
  arma::colvec theta_old(theta_dim);
  theta_old.subvec(0, piv_dim-1) = arma::conv_to<arma::colvec>::from(piv_old);
  theta_old.subvec(piv_dim, piv_dim+Pi_dim-1) = Pi_old_vec;
  //theta_old.subvec(piv_dim+Pi_dim, piv_dim+Pi_dim+Mu_dim-1) = Mu_old_vec;
  //theta_old.subvec(piv_dim+Pi_dim+Mu_dim, piv_dim+Pi_dim+Mu_dim+Si_dim-1) = Si_old_vec;

  bool lk_conv = (abs(lk - lk_old)/abs(lk_old) < tol_lk);
  bool theta_conv = (arma::max(arma::abs(theta - theta_old)) < tol_theta);
  bool maxit_reached = (it > maxit-1);
  bool minit_done = (it > 2);

  bool alt = (maxit_reached + (theta_conv && lk_conv)) && minit_done;

  return alt;
}



// [[Rcpp::export]]
Rcpp::List HMcont_E_step(int n, int TT, int k, Rcpp::List llk_list, arma::cube Pi) {
  arma::cube Psi = llk_list["Psi"];
  arma::cube L = llk_list["L"];

  arma::cube V(n, k, TT, arma::fill::zeros);
  arma::cube U(k, k, TT, arma::fill::zeros);
  arma::mat M(n, k, arma::fill::ones);
  arma::mat M_new(n, k);
  arma::mat Lt(n, k);
  arma::rowvec Tmpc(k);
  arma::rowvec Tmpr(k);
  arma::mat Tmp(k, k);
  
  Lt = L.slice(TT-1);
  Lt.each_col() /= arma::sum(L.slice(TT-1), 1);
  V.slice(TT-1) = Lt;
  for (int i = 0; i < n; i++) {
    Tmpc = L(arma::span(i), arma::span(), arma::span(TT-2));
    Tmpr = Psi(arma::span(i), arma::span(), arma::span(TT-1));
    Tmp = (Tmpc.t() * Tmpr) % Pi.slice(TT-1);
    U.slice(TT-1) += (Tmp/arma::accu(Tmp));
  }
  if (TT > 2) {
    for (int t = TT-2; t > 0; t--) {
      M_new = (Psi.slice(t+1) % M) * Pi.slice(t+1).t();
      M_new.each_col() /= arma::sum(M_new, 1);
      Lt = L.slice(t) % M_new;
      Lt.each_col() /= arma::sum(Lt, 1);
      V.slice(t) = Lt;
      for (int i = 0; i < n; i++) {
        Tmpc = L(arma::span(i), arma::span(), arma::span(t-1));
        Tmpr = Psi(arma::span(i), arma::span(), arma::span(t));
        Tmp = (Tmpc.t() * (Tmpr % M_new.row(i))) % Pi.slice(t);
        U.slice(t) += (Tmp/arma::accu(Tmp));
      }
      M = M_new;
    }
  }
  M_new = (Psi.slice(1) % M) * Pi.slice(1).t();
  M_new.each_col() /= arma::sum(M_new, 1);
  Lt = L.slice(0) % M_new;
  Lt.each_col() /= arma::sum(Lt, 1);
  V.slice(0) = Lt;

  return Rcpp::List::create(Rcpp::Named("V") = V,
                            Rcpp::Named("U") = U);
}



// [[Rcpp::export]]
Rcpp::List HMcont_TE_step(int n, int TT, int k, Rcpp::List llk_list, arma::cube Pi, double temp) {
  arma::cube Psi = llk_list["Psi"];
  arma::cube L = llk_list["L"];
  
  arma::cube V(n, k, TT, arma::fill::zeros);
  arma::cube U(k, k, TT, arma::fill::zeros);
  arma::mat M(n, k, arma::fill::ones);
  arma::mat M_new(n, k);
  arma::mat Lt(n, k);
  arma::rowvec Tmpc(k);
  arma::rowvec Tmpr(k);
  arma::mat Tmp(k, k);
  
  Lt = pow(L.slice(TT-1), 1/temp);
  Lt.each_col() /= arma::sum(L.slice(TT-1), 1);
  V.slice(TT-1) = Lt;
  for (int i = 0; i < n; i++) {
    Tmpc = L(arma::span(i), arma::span(), arma::span(TT-2));
    Tmpr = Psi(arma::span(i), arma::span(), arma::span(TT-1));
    Tmp = pow((Tmpc.t() * Tmpr) % Pi.slice(TT-1), 1/temp);
    U.slice(TT-1) += (Tmp/arma::accu(Tmp));
  }
  if (TT > 2) {
    for (int t = TT-2; t > 0; t--) {
      M_new = (Psi.slice(t+1) % M) * Pi.slice(t+1).t();
      M_new.each_col() /= arma::sum(M_new, 1);
      Lt = pow(L.slice(t) % M_new, 1/temp);
      Lt.each_col() /= arma::sum(Lt, 1);
      V.slice(t) = Lt;
      for (int i = 0; i < n; i++) {
        Tmpc = L(arma::span(i), arma::span(), arma::span(t-1));
        Tmpr = Psi(arma::span(i), arma::span(), arma::span(t));
        Tmp = pow((Tmpc.t() * (Tmpr % M_new.row(i))) % Pi.slice(t), 1/temp);
        U.slice(t) += (Tmp/arma::accu(Tmp));
      }
      M = M_new;
    }
  }
  M_new = (Psi.slice(1) % M) * Pi.slice(1).t();
  M_new.each_col() /= arma::sum(M_new, 1);
  Lt = pow(L.slice(0) % M_new, 1/temp);
  Lt.each_col() /= arma::sum(Lt, 1);
  V.slice(0) = Lt;
  
  return Rcpp::List::create(Rcpp::Named("U") = U,
                            Rcpp::Named("V") = V);
}



// [[Rcpp::export]]
Rcpp::List HMcont_M_step(arma::mat Sv, int n, int r, int TT, int k, Rcpp::List E_list, int modBasic) {
  arma::rowvec piv(k);
  arma::cube Pi(k, k, TT);
  arma::mat Mu(r, k);
  arma::mat Si(r, r);
  arma::cube V = E_list["V"];
  arma::cube U = E_list["U"];

  arma::cube V_aux(n, TT, k);
  for (int i = 0; i < n; i++) {
    for (int u = 0; u < k; u++) {
      for (int t = 0; t < TT; t++) {
        V_aux(i, t, u) = V(i, u, t);
      }
    }
  }
  arma::mat Vv = arma::reshape(arma::mat(V_aux.memptr(), V_aux.n_elem, 1, false), n*TT, k);
  arma::mat SiTmp(r, r, arma::fill::zeros);
  arma::mat Tmp(n*TT, r);
  arma::mat Tmp_mod(n*TT, r);
  arma::colvec ones_nTT(n*TT, arma::fill::ones);
  
  for (int u = 0; u < k; u++) {
    Mu.col(u) = (Sv.t() * Vv.col(u))/arma::accu(Vv.col(u));
    Tmp = Sv - ones_nTT * Mu.col(u).t();
    Tmp_mod = Tmp;
    for (int i = 0; i < n*TT; i++) {
      Tmp_mod.row(i) *= Vv(i, u);
    }
    SiTmp += (Tmp.t() * Tmp_mod);
  }
  Si = SiTmp/(n * TT);

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
                            Rcpp::Named("Mu") = Mu,
                            Rcpp::Named("Si") = Si);
}



// [[Rcpp::export]]
Rcpp::List HMcont_EM(arma::cube S, int k, double tol_lk, double tol_theta, int maxit, arma::rowvec piv, arma::cube Pi, arma::mat Mu, arma::mat Si, int modBasic) {
  arma::rowvec piv_old; arma::cube Pi_old; arma::mat Mu_old; arma::mat Si_old;
  
  Rcpp::List llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
  double lk = llk_list["LogLik"];
  double lk_old;
  arma::rowvec lkv(maxit);
  int it = 0;
  bool alt = false;

  int n = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), n*TT, r);
  arma::cube V(n, k, TT);

  while (!alt) {
    piv_old = piv; Pi_old = Pi; Mu_old = Mu; Si_old = Si;
    Rcpp::List E_list = HMcont_E_step(n, TT, k, llk_list, Pi);
    Rcpp::List M_list = HMcont_M_step(Sv, n, r, TT, k, E_list, modBasic);
    Rcpp::NumericMatrix piv_aux = M_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
    Rcpp::NumericVector Pi_aux = M_list["Pi"];
    Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
    Rcpp::NumericMatrix Mu_aux = M_list["Mu"];
    Mu = arma::mat(Mu_aux.begin(), r, k, false);
    Rcpp::NumericMatrix Si_aux = M_list["Si"];
    Si = arma::mat(Si_aux.begin(), r, r, false);
    Rcpp::NumericVector V_aux = E_list["V"];
    V = arma::cube(V_aux.begin(), n, k, TT, false);
    
    lk_old = lk;
    llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
    lk = llk_list["LogLik"];
    lkv(it) = lk;
    it++;
    alt = HMcont_CheckConvergence(lk, lk_old, piv, Pi, Mu, Si, piv_old, Pi_old, Mu_old, Si_old, it, tol_lk, tol_theta, maxit);
  }
  
  int np = (k - 1) + k * r + r * (r + 1)/2;
  if (modBasic == 0) {
    np += (TT - 1) * k * (k - 1);
  } else if (modBasic == 1) {
    np += k * (k - 1);
  }
    
  double aic = -2 * lk + np * 2;
  double bic = -2 * lk + np * log(n);
  
  return Rcpp::List::create(Rcpp::Named("lk") = lk,
                            Rcpp::Named("lkv") = lkv.head(it),
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Pi") = Pi,
                            Rcpp::Named("Mu") = Mu,
                            Rcpp::Named("Si") = Si,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("np") = np,
                            Rcpp::Named("modBasic") = modBasic,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List HMcont_TEM(arma::cube S, int k, double tol_lk, double tol_theta, int maxit, arma::rowvec piv, arma::cube Pi, arma::mat Mu, arma::mat Si, int modBasic, int profile, Rcpp::List profile_pars) {
  arma::rowvec piv_old; arma::cube Pi_old; arma::mat Mu_old; arma::mat Si_old;
  
  Rcpp::List llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
  double lk = llk_list["LogLik"];
  double lk_old;
  arma::rowvec lkv(maxit);
  int it = 0;
  bool alt = false;
  
  int n = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), n*TT, r);
  arma::cube V(n, k, TT);
  
  while (!alt) {
    double temp = HMcont_Temperature(it+1, profile, profile_pars);
    piv_old = piv; Pi_old = Pi; Mu_old = Mu; Si_old = Si;
    Rcpp::List tE_list = HMcont_TE_step(n, TT, k, llk_list, Pi, temp);
    Rcpp::List tM_list = HMcont_M_step(Sv, n, r, TT, k, tE_list, modBasic);
    Rcpp::NumericMatrix piv_aux = tM_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
    Rcpp::NumericVector Pi_aux = tM_list["Pi"];
    Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
    Rcpp::NumericMatrix Mu_aux = tM_list["Mu"];
    Mu = arma::mat(Mu_aux.begin(), r, k, false);
    Rcpp::NumericMatrix Si_aux = tM_list["Si"];
    Si = arma::mat(Si_aux.begin(), r, r, false);
    Rcpp::NumericVector V_aux = tE_list["V"];
    V = arma::cube(V_aux.begin(), n, k, TT, false);
  
    lk_old = lk;
    llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
    lk = llk_list["LogLik"];
    lkv(it) = lk;
    it++;
    alt = HMcont_CheckConvergence(lk, lk_old, piv, Pi, Mu, Si, piv_old, Pi_old, Mu_old, Si_old, it, tol_lk, tol_theta, maxit-1);
  }

  Rcpp::List E_list = HMcont_E_step(n, TT, k, llk_list, Pi);
  Rcpp::List M_list = HMcont_M_step(Sv, n, r, TT, k, E_list, modBasic);
  Rcpp::NumericMatrix piv_aux = M_list["piv"];
  piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
  Rcpp::NumericVector Pi_aux = M_list["Pi"];
  Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
  Rcpp::NumericMatrix Mu_aux = M_list["Mu"];
  Mu = arma::mat(Mu_aux.begin(), r, k, false);
  Rcpp::NumericMatrix Si_aux = M_list["Si"];
  Si = arma::mat(Si_aux.begin(), r, r, false);
  Rcpp::NumericVector V_aux = E_list["V"];
  V = arma::cube(V_aux.begin(), n, k, TT, false);
  
  lk_old = lk;
  llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
  lk = llk_list["LogLik"];
  lkv(it) = lk;
  it++;
  
  int np = (k - 1) + k * r + r * (r + 1)/2;
  if (modBasic == 0) {
    np += (TT - 1) * k * (k - 1);
  } else if (modBasic == 1) {
    np += k * (k - 1);
  }
  
  double aic = -2 * lk + np * 2;
  double bic = -2 * lk + np * log(n);

  return Rcpp::List::create(Rcpp::Named("lk") = lk,
                            Rcpp::Named("lkv") = lkv.head(it),
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Pi") = Pi,
                            Rcpp::Named("Mu") = Mu,
                            Rcpp::Named("Si") = Si,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("np") = np,
                            Rcpp::Named("modBasic") = modBasic,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic,
                            Rcpp::Named("profile") = profile,
                            Rcpp::Named("profile_pars") = profile_pars);
}



// [[Rcpp::export]]
Rcpp::List HMcont_Initialization_step (arma::cube S, int n, int r, int TT, int k, int n_parents) {
  Rcpp::List P(n_parents);
  
  for (int b = 0; b < n_parents; b++) {
    arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), n*TT, r);
    arma::mat Mu(r, k, arma::fill::zeros);
    arma::colvec mu = arma::mean(Sv.t(), 1);
    arma::mat Si = arma::cov(Sv);
    for (int u = 0; u < k; u++) {
      Mu.col(u) = arma::mvnrnd(mu, Si, 1);
    }
    arma::cube Pi(k, k, TT, arma::fill::randu);
    for (int t = 1; t < TT; t++) {
      Pi.slice(t) = arma::diagmat(1/arma::sum(Pi.slice(t), 1)) * Pi.slice(t);
    }
    Pi.slice(0) = arma::mat(k, k, arma::fill::zeros);
    arma::rowvec piv(k, arma::fill::randu);
    double piv_sum = arma::accu(piv);
    piv /= piv_sum;
    Rcpp::List llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
    arma::cube Psi = llk_list["Psi"];
    arma::cube L = llk_list["L"];
    
    arma::cube V(n, k, TT, arma::fill::zeros);
    arma::cube U(k, k, TT, arma::fill::zeros);
    arma::mat M(n, k, arma::fill::ones);
    arma::mat M_new(n, k);
    arma::mat Lt(n, k);
    arma::rowvec Tmpc(k);
    arma::rowvec Tmpr(k);
    arma::mat Tmp(k, k);
    
    Lt = L.slice(TT-1);
    Lt.each_col() /= arma::sum(L.slice(TT-1), 1);
    V.slice(TT-1) = Lt;
    for (int i = 0; i < n; i++) {
      Tmpc = L(arma::span(i), arma::span(), arma::span(TT-2));
      Tmpr = Psi(arma::span(i), arma::span(), arma::span(TT-1));
      Tmp = (Tmpc.t() * Tmpr) % Pi.slice(TT-1);
      U.slice(TT-1) += (Tmp/arma::accu(Tmp));
    }
    if (TT > 2) {
      for (int t = TT-2; t > 0; t--) {
        M_new = (Psi.slice(t+1) % M) * Pi.slice(t+1).t();
        M_new.each_col() /= arma::sum(M_new, 1);
        Lt = L.slice(t) % M_new;
        Lt.each_col() /= arma::sum(Lt, 1);
        V.slice(t) = Lt;
        for (int i = 0; i < n; i++) {
          Tmpc = L(arma::span(i), arma::span(), arma::span(t-1));
          Tmpr = Psi(arma::span(i), arma::span(), arma::span(t));
          Tmp = (Tmpc.t() * (Tmpr % M_new.row(i))) % Pi.slice(t);
          U.slice(t) += (Tmp/arma::accu(Tmp));
        }
        M = M_new;
      }
    }
    M_new = (Psi.slice(1) % M) * Pi.slice(1).t();
    M_new.each_col() /= arma::sum(M_new, 1);
    Lt = L.slice(0) % M_new;
    Lt.each_col() /= arma::sum(Lt, 1);
    V.slice(0) = Lt;
    
    P(b) = Rcpp::List::create(Rcpp::Named("V") = V, Rcpp::Named("U") = U);
  }
  
  return P;
}



// [[Rcpp::export]]
Rcpp::List HMcont_ME_step(arma::cube S, int k, double tol_lk, double tol_theta, int maxit, Rcpp::List P, int modBasic) {
  int n = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), n*TT, r);
  arma::cube V(n, k, TT);
  arma::cube U(k, k, TT);

  arma::rowvec piv(k, arma::fill::zeros); arma::rowvec piv_old(k);
  arma::cube Pi(k, k, TT, arma::fill::zeros); arma::cube Pi_old(k, k, TT);
  arma::mat Mu(r, k, arma::fill::zeros); arma::mat Mu_old(r, k);
  arma::mat Si(r, r, arma::fill::zeros); arma::mat Si_old(r, r);
  double lk = 0; double lk_old;
  int it = 0;
  bool alt = false;
    
  while (!alt) {
    lk_old = lk; piv_old = piv; Pi_old = Pi; Mu_old = Mu; Si_old = Si;
    
    Rcpp::List M_list = HMcont_M_step(Sv, n, r, TT, k, P, modBasic);
    Rcpp::NumericMatrix piv_aux = M_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
    Rcpp::NumericVector Pi_aux = M_list["Pi"];
    Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
    Rcpp::NumericMatrix Mu_aux = M_list["Mu"];
    Mu = arma::mat(Mu_aux.begin(), r, k, false);
    Rcpp::NumericMatrix Si_aux = M_list["Si"];
    Si = arma::mat(Si_aux.begin(), r, r, false);
    Rcpp::List llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
    lk = llk_list["LogLik"];
    Rcpp::List E_list = HMcont_E_step(n, TT, k, llk_list, Pi);
    Rcpp::NumericVector V_aux = E_list["V"];
    V = arma::cube(V_aux.begin(), n, k, TT, false);
    Rcpp::NumericVector U_aux = E_list["U"];
    U = arma::cube(U_aux.begin(), k, k, TT, false);
    P["V"] = V; P["U"] = U;

    it++;
    alt = HMcont_CheckConvergence(lk, lk_old, piv, Pi, Mu, Si, piv_old, Pi_old, Mu_old, Si_old, it, tol_lk, tol_theta, maxit);
  }

  return Rcpp::List::create(Rcpp::Named("V") = V,
                            Rcpp::Named("U") = U,
                            Rcpp::Named("fit") = lk);
}



// [[Rcpp::export]]
Rcpp::List HMcont_LastME_step (arma::cube S, int k, double tol_lk, double tol_theta, int maxit, Rcpp::List P, int modBasic) {
  int n = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;
  arma::mat Sv = arma::reshape(arma::mat(S.memptr(), S.n_elem, 1, false), n*TT, r);
  arma::cube V(n, k, TT);
  arma::cube U(k, k, TT);
  
  arma::rowvec piv(k, arma::fill::zeros); arma::rowvec piv_old(k);
  arma::cube Pi(k, k, TT, arma::fill::zeros); arma::cube Pi_old(k, k, TT);
  arma::mat Mu(r, k, arma::fill::zeros); arma::mat Mu_old(r, k);
  arma::mat Si(r, r, arma::fill::zeros); arma::mat Si_old(r, r);
  double lk = 0; double lk_old;
  int it = 0;
  bool alt = false;
  
  while (!alt) {
    lk_old = lk; piv_old = piv; Pi_old = Pi; Mu_old = Mu; Si_old = Si;
    
    Rcpp::List M_list = HMcont_M_step(Sv, n, r, TT, k, P, modBasic);
    Rcpp::NumericMatrix piv_aux = M_list["piv"];
    piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
    Rcpp::NumericVector Pi_aux = M_list["Pi"];
    Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
    Rcpp::NumericMatrix Mu_aux = M_list["Mu"];
    Mu = arma::mat(Mu_aux.begin(), r, k, false);
    Rcpp::NumericMatrix Si_aux = M_list["Si"];
    Si = arma::mat(Si_aux.begin(), r, r, false);
    Rcpp::List llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
    lk = llk_list["LogLik"];
    Rcpp::List E_list = HMcont_E_step(n, TT, k, llk_list, Pi);
    Rcpp::NumericVector V_aux = E_list["V"];
    V = arma::cube(V_aux.begin(), n, k, TT, false);
    Rcpp::NumericVector U_aux = E_list["U"];
    U = arma::cube(U_aux.begin(), k, k, TT, false);
    P["V"] = V; P["U"] = U;
    
    it++;
    alt = HMcont_CheckConvergence(lk, lk_old, piv, Pi, Mu, Si, piv_old, Pi_old, Mu_old, Si_old, it, tol_lk, tol_theta, maxit);
  }
  
  // Last M Step
  Rcpp::List M_list = HMcont_M_step(Sv, n, r, TT, k, P, modBasic);
  Rcpp::NumericMatrix piv_aux = M_list["piv"];
  piv = arma::rowvec(piv_aux.begin(), piv_aux.length(), false);
  Rcpp::NumericVector Pi_aux = M_list["Pi"];
  Pi = arma::cube(Pi_aux.begin(), k, k, TT, false);
  Rcpp::NumericMatrix Mu_aux = M_list["Mu"];
  Mu = arma::mat(Mu_aux.begin(), r, k, false);
  Rcpp::NumericMatrix Si_aux = M_list["Si"];
  Si = arma::mat(Si_aux.begin(), r, r, false);
  Rcpp::List llk_list = HMcont_ComputeLogLik(S, piv, Pi, Mu, Si);
  lk = llk_list["LogLik"];
  
  int np = (k - 1) + k * r + r * (r + 1)/2;
  if (modBasic == 0) {
    np += (TT - 1) * k * (k - 1);
  } else if (modBasic == 1) {
    np += k * (k - 1);
  }
  
  double aic = -2 * lk + np * 2;
  double bic = -2 * lk + np * log(n);
  
  return Rcpp::List::create(Rcpp::Named("lk") = lk,
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("Pi") = Pi,
                            Rcpp::Named("Mu") = Mu,
                            Rcpp::Named("Si") = Si,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("np") = np,
                            Rcpp::Named("modBasic") = modBasic,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("bic") = bic);
}



// [[Rcpp::export]]
Rcpp::List HMcont_CrossOver_step (int n, int TT, Rcpp::List P, int n_children, int n_parents) {
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
      int i = arma::randi(arma::distr_param(0, n-1));
      arma::cube V_aux = V1(arma::span(i, n-1), arma::span(), arma::span(t));
      V1(arma::span(i, n-1), arma::span(), arma::span(t)) = V2(arma::span(i, n-1), arma::span(), arma::span(t));
      V2(arma::span(i, n-1), arma::span(), arma::span(t)) = V_aux;
    }

    V_children(h) = Rcpp::List::create(Rcpp::Named("V") = V1, Rcpp::Named("U") = U1);
    V_children(h+1) = Rcpp::List::create(Rcpp::Named("V") = V2, Rcpp::Named("U") = U2);
    h += 2;
  }

  return V_children;
}



// [[Rcpp::export]]
Rcpp::List HMcont_Selection_step (Rcpp::List PV_p, Rcpp::List PV_c, arma::rowvec fit_p, arma::rowvec fit_c, int n_parents, int n_children) {
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
Rcpp::List HMcont_Mutation_step (int n, int TT, int k, Rcpp::List P, double prob_mut) {
  arma::cube V = P["V"];
  arma::cube U = P["U"];

  for (int t = 0; t < TT; t++) {
    for (int i = 0; i < n; i++) {
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
Rcpp::List HMcont_EEM(arma::cube S, int k, double tol_lk, double tol_theta, int maxit, int modBasic, int n_parents, int n_children, double prob_mut, int R) {
  int n = S.n_rows;
  int TT = S.n_cols;
  int r = S.n_slices;

  // 1. Initial values
  Rcpp::List PV1 = HMcont_Initialization_step(S = S, n = n, r = r, TT = TT, k = k, n_parents = n_parents);
  Rcpp::List PV2(n_parents), PV3(n_children), PV4(n_children), PV5(n_parents), PV6(n_parents);
  arma::rowvec fit2(n_parents), fit4(n_children), fit5(n_parents);
  double fit_old;
  bool conv = false;
  int it = 0;
  
  while (!conv) {
    // 2. Update parents and compute fitness
    for (int b = 0; b < n_parents; b++) {
      Rcpp::List P_aux = PV1[b];
      Rcpp::List PV2_aux = HMcont_ME_step(S, k, tol_lk, tol_theta, R, P_aux, modBasic);
      PV2(b) = Rcpp::List::create(Rcpp::Named("V") = PV2_aux["V"], Rcpp::Named("U") = PV2_aux["U"]);
      fit2(b) = PV2_aux["fit"];
    }
    // 3. Cross-over
    PV3 = HMcont_CrossOver_step(n, TT, PV2, n_children, n_parents);
    // 4. Update children and compute fitness
    for (int b = 0; b < n_children; b++) {
      Rcpp::List P_aux = PV3[b];
      Rcpp::List PV4_aux = HMcont_ME_step(S, k, tol_lk, tol_theta, R, P_aux, modBasic);
      PV4(b) = Rcpp::List::create(Rcpp::Named("V") = PV4_aux["V"], Rcpp::Named("U") = PV4_aux["U"]);
      fit4(b) = PV4_aux["fit"];
    }
    // 5. Select new parents
    PV5 = HMcont_Selection_step(PV2, PV4, fit2, fit4, n_parents, n_children);
    Rcpp::NumericVector fit_aux = PV5["fit"];
    fit5 = arma::rowvec(fit_aux.begin(), fit_aux.length(), false);
    PV5 = PV5["PV"];
    // 6. Mutation
    PV6(0) = PV5(0);
    for (int b = 1; b < n_parents; b++) {
      Rcpp::List P_aux = PV5[b];
      PV6(b) = HMcont_Mutation_step(n, TT, k, P_aux, prob_mut);
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
  Rcpp::List out = HMcont_LastME_step(S, k, tol_lk, tol_theta, maxit, P_aux, modBasic);

  return out;
}







