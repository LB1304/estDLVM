#include <RcppArmadillo.h>
#include <RcppParallel.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace RcppParallel;



// [[Rcpp::export]]
int bin_coeff(const int n, const int m) {
  if (m == 0) {
    return 1;
  }
  
  int step1 = n - m + 1;
  int step0;
  for (int i = 1; i < m; ++i) {
    step1 = (step0 = step1) * (n - m + 1 + i) / (i + 1);
  }
  
  return step1;
}



// [[Rcpp::export]]
bool SB_CheckConvergence (double LogLik, double LogLik_old, arma::rowvec piv, arma::mat B, arma::rowvec piv_old, arma::mat B_old, int it, double tol_lk, double tol_theta, int maxit) {
  int piv_dim = piv.n_elem;
  int B_dim = B.n_elem;
  int theta_dim = piv_dim + B_dim;
  
  arma::colvec B_vec(B.memptr(), B_dim, false);
  arma::colvec theta(theta_dim);
  theta.subvec(0, piv_dim-1) = arma::conv_to<arma::colvec>::from(piv);
  theta.subvec(piv_dim, piv_dim+B_dim-1) = B_vec;
  
  arma::colvec B_old_vec(B_old.memptr(), B_dim, false);
  arma::colvec theta_old(theta_dim);
  theta_old.subvec(0, piv_dim-1) = arma::conv_to<arma::colvec>::from(piv_old);
  theta_old.subvec(piv_dim, piv_dim+B_dim-1) = B_old_vec;
  
  bool LogLik_conv = (abs(LogLik - LogLik_old)/abs(LogLik_old) < tol_lk);
  bool theta_conv = (arma::max(arma::abs(theta - theta_old)) < tol_theta);
  bool maxit_reached = (it > maxit-1);
  bool minit_done = (it > 2);
  
  bool alt = (maxit_reached + (theta_conv && LogLik_conv)) && minit_done;
  
  return alt;
}



// [[Rcpp::export]]
Rcpp::List SB_k1 (arma::mat Y) {
  int n = Y.n_rows;
  
  double n_edges = (arma::accu(Y.diag()) + arma::accu(Y))/2;
  double piv = 1.0;
  double B = n_edges/bin_coeff(n+2-1, 2);
  
  arma::mat V(n, 1, arma::fill::ones);
  
  double log_B1, log_B2;
  if (B == 0.0 || B == 1.0) {
    log_B1 = 0.0, log_B2 = 0.0;
  } else {
    log_B1 = log(B), log_B2 = log(1 - B);
  }
  double LogLik = n_edges * log_B1 + (bin_coeff(n+2-1, 2) - n_edges) * log_B2;
  double J = n_edges * log_B1 + (bin_coeff(n+2-1, 2) - n_edges) * log_B2;
  double ICL = LogLik - 0.5 * log(bin_coeff(n, 2));
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("J") = J,
                            Rcpp::Named("J_vec") = J,
                            Rcpp::Named("it") = 1,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("B") = B,
                            Rcpp::Named("k") = 1,
                            Rcpp::Named("N_par") = 1,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("icl") = ICL);
}



// [[Rcpp::export]]
arma::mat SB_compute_tau (arma::mat V, arma::mat B, arma::rowvec piv, arma::mat Y, int k, int n) {
  arma::mat V_new(n, k);
  
  for (int i = 0; i < n; i++) {
    arma::colvec y = Y.col(i);
    for (int q = 0; q < k; q++) {
      arma::rowvec b = B.row(q);
      arma::uvec ind_b = arma::find(b == 0 || b == 1);
      
      arma::rowvec log_b1 = log(b);
      log_b1.elem(ind_b).zeros();
      arma::rowvec log_b2 = log(1-b);
      log_b2.elem(ind_b).zeros();
      
      arma::mat BY1 = y*log_b1;
      arma::mat BY2 = (1-y)*log_b2;
      arma::mat BY = V % (BY1+BY2);
      V_new(i, q) = arma::accu(BY) + log(piv(q));
    }
  }
  
  for (int i = 0; i < n; i++) {
    double V_i_max = arma::max(V_new.row(i));
    for (int q = 0; q < k; q++) {
      V_new(i, q) = exp(V_new(i, q) - V_i_max);
    }
    double rsum = arma::sum(V_new.row(i));
    arma::mat V_sum = arma::ones<arma::mat>(n, k);
    V_sum.row(i).fill(1/rsum);
    V_new %= V_sum;
  }
  
  return V_new;
}



// [[Rcpp::export]]
Rcpp::List SB_VE_step (arma::mat V, arma::mat B, arma::rowvec piv, arma::mat Y, int k, int n, double tol_lk, int maxit_FP) {
  arma::mat V_new(n, k);
  
  bool alt = false;
  int it = 0;
  while (!alt) {
    it += 1;
    V_new = SB_compute_tau(V, B, piv, Y, k, n);
    arma::mat diff_taus = abs(V_new - V);
    bool converged = (diff_taus.max() < tol_lk);
    bool maxit_reached = (it > maxit_FP-1);
    alt = converged + maxit_reached;
    V = V_new;
  }
  
  return Rcpp::List::create(Rcpp::Named("V") = V_new, Rcpp::Named("it") = it);
}



// [[Rcpp::export]]
arma::mat SB_compute_B (arma::mat V, arma::mat Y, int k, int n) {
  arma::mat B(k, k);
  
  for (int q = 0; q < k; q++) {
    arma::colvec q1 = V.col(q);
    for (int p = q; p < k; p++) {
      arma::rowvec q2 = arma::conv_to<arma::rowvec>::from(V.col(p));
      arma::mat den = q1*q2;
      double den_sum = arma::accu(den);
      arma::mat num = den % Y;
      double num_sum = arma::accu(num);
      double aux = 0.0;
      if (den_sum != 0) {
        aux = num_sum/den_sum;
      }
      B(q, p) = aux;
      B(p, q) = aux;
    }
  }
  
  return B;
}



// [[Rcpp::export]]
arma::rowvec SB_compute_pi (arma::mat V, int n){
  arma::rowvec piv = sum(V, 0)/n;
  
  return piv;
}



// [[Rcpp::export]]
double SB_compute_ELBO (arma::mat V, arma::mat B, arma::rowvec piv, arma::mat Y, int k, int n) {
  arma::mat log_tau = log(V);
  arma::uvec ind_log_tau = arma::find(V == 0);
  log_tau.elem(ind_log_tau) = arma::zeros<arma::vec>(ind_log_tau.n_elem);
  arma::rowvec log_pi = log(piv);
  arma::uvec ind_log_pi = arma::find(piv == 0);
  log_pi.elem(ind_log_pi) = arma::zeros<arma::vec>(ind_log_pi.n_elem);
  
  arma::mat log_Pi(n, k);
  for (int i = 0; i < n; i++) {
    log_Pi.row(i) = log_pi;
  }
  double ELBO = accu(V % log_Pi - V % log_tau);
  
  arma::mat J(n, k);
  
  for (int i = 0; i < n; i++) {      
    arma::colvec y = Y.col(i);
    for (int q = 0; q < k; q++) {
      arma::rowvec b = B.row(q);
      arma::uvec ind_b = arma::find(b == 0 || b == 1);
      
      arma::rowvec log_b1 = log(b);
      log_b1.elem(ind_b).zeros();
      arma::rowvec log_b2 = log(1-b);
      log_b2.elem(ind_b).zeros();
      
      arma::mat BY1 = y*log_b1;
      arma::mat BY2 = (1-y)*log_b2;
      arma::mat BY = V % (BY1+BY2);
      J(i, q) = V(i, q) * arma::accu(BY);
    }
  }
  
  ELBO += arma::accu(J);
  
  return ELBO;
}



// [[Rcpp::export]]
double SB_compute_LogLik (arma::mat V, arma::mat B, arma::rowvec piv, arma::mat Y, int k, int n) {
  // Creo Z come modifica "hard" di V (0,1)
  arma::mat Z(n, k);
  for (int i = 0; i < n; i++) {
    arma::rowvec V_i = V.row(i);
    arma::uword z_i = V_i.index_max();
    Z(i, z_i) = 1;
  }
  
  arma::rowvec log_pi = log(piv);
  arma::uvec ind_log_pi = arma::find(piv == 0);
  log_pi.elem(ind_log_pi) = arma::zeros<arma::vec>(ind_log_pi.n_elem);
  
  arma::mat log_Pi(n, k);
  for (int i = 0; i < n; i++) {
    log_Pi.row(i) = log_pi;
  }
  double LogLik = accu(Z % log_Pi);
  
  arma::mat J(n, k);
  
  for (int i = 0; i < n; i++) {      
    arma::colvec y = Y.col(i);
    for (int q = 0; q < k; q++) {
      arma::rowvec b = B.row(q);
      arma::uvec ind_b = arma::find(b == 0 || b == 1);
      
      arma::rowvec log_b1 = log(b);
      log_b1.elem(ind_b).zeros();
      arma::rowvec log_b2 = log(1-b);
      log_b2.elem(ind_b).zeros();
      
      arma::mat BY1 = y*log_b1;
      arma::mat BY2 = (1-y)*log_b2;
      arma::mat BY = V % (BY1+BY2);
      J(i, q) = V(i, q) * arma::accu(BY);
    }
  }
  
  LogLik += arma::accu(J);
  
  return LogLik;
}



// [[Rcpp::export]]
Rcpp::List SB_VEM (arma::mat Y, int k, double tol_lk, double tol_theta, int maxit, int maxit_FP, arma::mat V) {
  double J;
  double J_old = R_NegInf;
  arma::vec J_vec(maxit);
  int it = 0;
  bool alt = false;
  
  int n = Y.n_rows;
  arma::rowvec piv(k); arma::mat B(k, k); arma::mat V_new(n, k);
  
  while (!alt) {
    it++;
    
    // 1. M-Step
    B = SB_compute_B(V, Y, k, n);
    piv = SB_compute_pi(V, n);
    
    // 2. VE-Step
    Rcpp::List VE = SB_VE_step(V, B, piv, Y, k, n, tol_lk, maxit_FP);
    arma::mat V_aux = VE["V"];
    V_new = V_aux;
    int it_FP = VE["it"];
    
    // 3. Compute ELBO
    J = SB_compute_ELBO(V_new, B, piv, Y, k, n);
    J_vec(it-1) = J;
    
    // 4. Update convergence conditions
    bool converged = (abs(J - J_old)/abs(J_old) < tol_lk);
    bool maxit_reached = (it > maxit-1);
    bool fp_converged = (it_FP == 1);
    alt = maxit_reached + (converged * fp_converged);
    
    // 5. Update parameters
    V = V_new;
    J_old = J;
  }
  
  // 6. Additional M-step
  B = SB_compute_B(V, Y, k, n);
  piv = SB_compute_pi(V, n);
    
  // 7. ICL criterion
  double LogLik = SB_compute_LogLik(V, B, piv, Y, k, n);
  int n_B_par = bin_coeff(k+2-1, 2);
  int n_pi_par = k-1;
  double ICL_B_par = 0.5 * n_B_par * log(bin_coeff(n, 2));
  double ICL_pi_par = 0.5 * n_pi_par * log(n);
  double ICL = LogLik - ICL_pi_par - ICL_B_par;
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("J") = J,
                            Rcpp::Named("J_vec") = J_vec.head(it-1), 
                            Rcpp::Named("it") = it, 
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("B") = B,
                            Rcpp::Named("k") = k, 
                            Rcpp::Named("N_par") = n_pi_par + n_B_par,
                            Rcpp::Named("V") = V, 
                            Rcpp::Named("icl") = ICL);
}



// [[Rcpp::export]]
Rcpp::List SB_Initialization_step (int n, int k, int n_parents) {
  Rcpp::List P(n_parents);
  
  for (int b = 0; b < n_parents; b++) {
    arma::mat V = arma::randu(n, k);
    arma::mat V_sum = arma::ones<arma::mat>(n, k);
    for (int i = 0; i < n; i++) {
      double rsum = arma::sum(V.row(i));
      V_sum.row(i).fill(rsum);
    }
    V /= V_sum;
    
    P(b) = V;
  }
  
  return P;
}



// [[Rcpp::export]]
Rcpp::List SB_ME_step (arma::mat Y, int n, arma::mat V, int k, int R, int maxit_FP, double tol_lk, double tol_theta) {
  arma::mat B(k, k);
  arma::rowvec piv(k);
  arma::mat V_new(n, k);
  double J;
  
  bool alt = false;
  double J_old = R_NegInf;
  int it = 0;
  while (!alt) {
    it +=1;
    
    // 1. M-Step
    B = SB_compute_B(V, Y, k, n);
    piv = SB_compute_pi(V, n);
    
    // 2. VE-Step
    Rcpp::List VE = SB_VE_step(V, B, piv, Y, k, n, tol_lk, maxit_FP);
    arma::mat V_aux = VE["V"];
    V_new = V_aux;
    int it_FP = VE["it"];
    
    // 3. Compute ELBO
    J = SB_compute_ELBO(V_new, B, piv, Y, k, n);
    
    // 4. Update convergence conditions
    bool converged = (abs(J - J_old)/abs(J_old) < tol_lk);
    bool maxit_reached = (it > R-1);
    bool fp_converged = (it_FP == 1);
    alt = maxit_reached + (converged * fp_converged);
    
    // 5. Update parameters
    V = V_new;
    J_old = J;
  }
  
  return Rcpp::List::create(Rcpp::Named("V") = V_new,
                            Rcpp::Named("fit") = J,
                            Rcpp::Named("B") = B);
}



// [[Rcpp::export]]
Rcpp::List SB_LastME_step (arma::mat Y, int n, arma::mat V, int k, int R, int maxit_FP, double tol_lk, double tol_theta) {
  arma::mat B(k, k);
  arma::rowvec piv(k);
  arma::mat V_new(n, k);
  double J;
  
  bool alt = false;
  double J_old = R_NegInf;
  int it = 0;
  while (!alt) {
    it +=1;
    
    // 1. M-Step
    B = SB_compute_B(V, Y, k, n);
    piv = SB_compute_pi(V, n);
    
    // 2. VE-Step
    Rcpp::List VE = SB_VE_step(V, B, piv, Y, k, n, tol_lk, maxit_FP);
    arma::mat V_aux = VE["V"];
    V_new = V_aux;
    int it_FP = VE["it"];
    
    // 3. Compute ELBO
    J = SB_compute_ELBO(V_new, B, piv, Y, k, n);
    
    // 4. Update convergence conditions
    bool converged = (abs(J - J_old)/abs(J_old) < tol_lk);
    bool maxit_reached = (it > R-1);
    bool fp_converged = (it_FP == 1);
    alt = maxit_reached + (converged * fp_converged);
    
    // 5. Update parameters
    V = V_new;
    J_old = J;
  }
  
  // Last M step
  B = SB_compute_B(V, Y, k, n);
  piv = SB_compute_pi(V, n);
  
  J = SB_compute_ELBO(V, B, piv, Y, k, n);
  double LogLik = SB_compute_LogLik(V, B, piv, Y, k, n);
  int n_B_par = bin_coeff(k+2-1, 2);
  int n_pi_par = k-1;
  double ICL_B_par = 0.5 * n_B_par * log(bin_coeff(n, 2));
  double ICL_pi_par = 0.5 * n_pi_par * log(n);
  double ICL = LogLik - ICL_pi_par - ICL_B_par;
  
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik,
                            Rcpp::Named("J") = J,
                            Rcpp::Named("piv") = piv,
                            Rcpp::Named("B") = B,
                            Rcpp::Named("k") = k,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("N_par") = n_pi_par + n_B_par,
                            Rcpp::Named("icl") = ICL);
}



// [[Rcpp::export]]
Rcpp::List SB_CrossOver_step (int n, Rcpp::List P, int n_children, int n_parents) {
  Rcpp::List V_children(n_children);
  
  int h = 0;
  for (int b = 0; b < n_children/2; b++) {
    int ind1 = arma::randi(arma::distr_param(0, n_parents-1));
    arma::mat V1 = P[ind1];
    int ind2 = arma::randi(arma::distr_param(0, n_parents-1));
    arma::mat V2 = P[ind2];
    
    int i = arma::randi(arma::distr_param(0, n-1));
    
    arma::mat V_aux = V1.rows(i, n-1);
    V1.rows(i, n-1) = V2.rows(i, n-1);
    V2.rows(i, n-1) = V_aux;
    
    V_children(h) = V1;
    V_children(h+1) = V2;
    h += 2;
  }
  
  return V_children;
}



// [[Rcpp::export]]
Rcpp::List SB_Selection_step (Rcpp::List PV_p, Rcpp::List PV_c, arma::rowvec fit_p, arma::rowvec fit_c, int n_parents, int n_children) {
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
arma::mat SB_Mutation_step (int n, int k, arma::mat V, double prob_mut) {
  for (int i = 0; i < n; i++) {
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
Rcpp::List SB_EVEM (arma::mat Y, int k, double tol_lk, double tol_theta, int maxit, int maxit_FP, int n_parents, int n_children, double prob_mut, int R) {
  int n = Y.n_rows;
  
  // 1. Initial values
  Rcpp::List PV1 = SB_Initialization_step(n, k, n_parents);
  Rcpp::List PV2(n_parents), PV3(n_children), PV4(n_children), PV5(n_parents), PV6(n_parents);
  arma::rowvec fit2(n_parents), fit4(n_children), fit5(n_parents);
  double fit_old;
  arma::mat V(n, k);
  bool conv = false;
  int it = 0;
  
  while (!conv) {
    // 2. Update parents and compute fitness
    for (int b = 0; b < n_parents; b++) {
      Rcpp::NumericMatrix V_aux = PV1[b];
      V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
      Rcpp::List PV2_aux = SB_ME_step(Y, n, V, k, R, maxit_FP, tol_lk, tol_theta);
      PV2(b) = PV2_aux["V"];
      fit2(b) = PV2_aux["fit"];
    }
    // 3. Cross-over
    PV3 = SB_CrossOver_step(n, PV2, n_children, n_parents);
    // 4. Update children and compute fitness
    for (int b = 0; b < n_children; b++) {
      Rcpp::NumericMatrix V_aux = PV3[b];
      V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
      Rcpp::List PV4_aux = SB_ME_step(Y, n, V, k, R, maxit_FP, tol_lk, tol_theta);
      PV4(b) = PV4_aux["V"];
      fit4(b) = PV4_aux["fit"];
    }
    // 5. Select new parents
    PV5 = SB_Selection_step(PV2, PV4, fit2, fit4, n_parents, n_children);
    Rcpp::NumericVector fit_aux = PV5["fit"];
    fit5 = arma::rowvec(fit_aux.begin(), fit_aux.length(), false);
    PV5 = PV5["PV"];
    // 6. Mutation
    PV6(0) = PV5(0);
    for (int b = 1; b < n_parents; b++) {
      Rcpp::NumericMatrix V_aux = PV5[b];
      V = arma::mat(V_aux.begin(), V_aux.nrow(), V_aux.ncol(), false);
      PV6(b) = SB_Mutation_step(n, k, V, prob_mut);
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
  Rcpp::List out = SB_LastME_step(Y, n, V, k, R, maxit_FP, tol_lk, tol_theta);
  
  return out;
}



