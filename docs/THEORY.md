# Mathematical specification of the public Gaussian experiment

This is an analytic reference model, not a new claim about arbitrary neural
reasoning. It separates true statistical risk from computation error. Related
Gaussian inference and recurrent graph work is listed in [REFERENCES.md](REFERENCES.md).

## Variables and implementation assumptions

All modeled values are normalized and dimensionless (SI unit 1). They are not
physical bits, seconds, energy, or semantic information units. The implementation
uses float64, no quantization/zero-point, and single-threaded numerical libraries.

| Symbol | Meaning (日本語) | SI unit | Definition | Domain / assumptions | Type |
|---|---|---:|---|---|---|
| n,m | 状態数・観測数 | 1 | 49 and 48 | positive integers | integer scalars |
| i,j,q | ノード添字・出力位置 | 1 | q has no observation | 0 to n−1 | integers |
| E | 辺集合 | 1 | undirected graph edges | connected in the protocol | set |
| e_i,I | 座標基底・単位行列 | 1 | coordinate selector and identity | compatible dimensions | vector / matrix |
| a_ij,s_ij | 辺の重み・符号 | 1 | magnitude and relation sign | a>0; s is −1 or +1 | scalars |
| A | 符号付き隣接行列 | 1 | A_ij=a_ij s_ij on edges | symmetric, zero diagonal | matrix |
| tau | 事前の正則化 | 1 | positive diagonal contribution | >0 | scalar |
| Q,Sigma | 事前精度・共分散 | 1 | precision below, Sigma=Q inverse | positive definite | matrices |
| theta | 真の潜在状態 | 1 | zero-mean Gaussian with covariance Sigma | real n components | random vector |
| P | 観測選択行列 | 1 | selects every node except q | m by n | matrix |
| eta,R | 観測ノイズ・共分散 | 1 | independent zero-mean Gaussian noise | R positive diagonal | random vector / matrix |
| y | 観測値 | 1 | P theta + eta | fixed during inference | random vector |
| M,b | 事後精度・右辺 | 1 | defined below | M positive definite | matrix / vector |
| mu,Pi | 事後平均・共分散 | 1 | M inverse times b; M inverse | conditioned on y | vector / matrix |
| D,T,U | 局所反復の係数 | 1 | diagonal, transition, input projection | defined below | matrices |
| z_t,t | 中間状態・更新回数 | 1 | zero initialization and t updates | t is a nonnegative integer | vector / integer |
| gamma | 収縮率の上界 | 1 | maximum absolute row sum of T | <1 | scalar |
| C,k,v | 観測共分散・相互共分散・出力分散 | 1 | Cov(y), Cov(y,theta_q), Sigma_qq | C positive definite; v>0 | matrix / vector / scalar |
| w_t,w_star | 有限反復・最適出力の係数 | 1 | z_t,q=w_t^T y, mu_q=w_star^T y | m components | vectors |
| g | 任意の推定器 | 1 | measurable function of y | finite second moment | function |
| R_g,R_star | 真の二乗誤差・最小二乗誤差 | 1 | expectations defined below | nonnegative | scalars |
| Y_L,Y_E | 近傍・追加外部の観測 | 1 | partition of y | nonempty subsets for section 5 | random vectors |
| C_LL,C_EL,C_LE,C_EE | 観測共分散の部分行列 | 1 | corresponding blocks of C | C_LE=C_EL transpose | matrices |
| k_L,k_E | 出力との相互共分散 | 1 | corresponding blocks of k | compatible dimensions | vectors |
| c,S | 追加情報の条件付き共分散 | 1 | defined in section 5 | S positive definite | vector / matrix |
| R_L,R_LE | 近傍・全観測での最小MSE | 1 | conditional variances | nonnegative | scalars |
| rho,p_acc | 出力と予測の相関・符号正答率 | 1 | defined in section 6 | rho in [−1,1], probability in [0,1] | scalar / probability |
| Z_1,Z_2,phi | 標準正規と角度 | 1 (angle in rad) | independent normals; phi=arccos(rho) | phi in [0,pi] | random scalars / scalar |

## 1. Positive-definite generative model

The prior precision is

$$Q=\tau I+\sum_{(i,j)\in E}a_{ij}(e_i-s_{ij}e_j)(e_i-s_{ij}e_j)^T,$$

$$\Sigma=Q^{-1},\qquad \theta\sim N(0,\Sigma),\qquad y=P\theta+\eta,\quad\eta\sim N(0,R).$$

For any nonzero real vector, its quadratic form with Q equals tau times its
squared norm plus a sum of nonnegative weighted squared relation differences.
Because tau is strictly positive, Q is positive definite and invertible. R is
positive diagonal and independent of theta. No individual query truth is input
to an inference policy. The known prior/noise law is a strong modeling assumption.

## 2. Exact Bayesian target

Multiplying Gaussian prior and likelihood gives a negative log density whose
part depending on theta, after multiplication by two, is

$$\theta^TQ\theta+(y-P\theta)^TR^{-1}(y-P\theta).$$

Expanding the square, define

$$M=Q+P^TR^{-1}P,\qquad b=P^TR^{-1}y.$$

The theta-dependent terms become theta-transpose M theta minus twice theta-
transpose b. M is positive definite because Q is positive definite and the
added matrix is positive semidefinite. Completing the square with

$$\mu=M^{-1}b,\qquad\Pi=M^{-1}$$

gives a Gaussian posterior with mean mu and covariance Pi. Thus the query's
all-observation Bayes MSE is Pi_qq. This is the true generating distribution's
posterior, not merely a fitted model's internally consistent answer.

## 3. Local recurrence, reach and convergence

Set

$$D=\operatorname{diag}(M),\qquad T=D^{-1}(D-M),\qquad U=D^{-1}P^TR^{-1},$$

$$z_{t+1}=Tz_t+Uy,\qquad z_0=0.$$

Each node reads only neighboring previous states and its own observation. The
query's input row is zero. Every diagonal entry of M contains the corresponding
sum of absolute edge weights, plus positive tau and nonnegative observation
precision. Therefore each absolute row sum of T is strictly below one; its
maximum is gamma<1.

The posterior mean satisfies mu=T mu+Uy. Subtracting its equation from the
recurrence and applying induction yields

$$z_t-\mu=T^t(z_0-\mu),\qquad\|z_t-\mu\|_\infty\le\gamma^t\|z_0-\mu\|_\infty.$$

The right side tends to zero, so the iteration converges to the Bayes mean.
This norm bound does not imply that every query's risk or classification
accuracy improves at each intermediate step.

The first update places observations into their own nodes; the unobserved query
remains zero. In each later synchronous update a dependency can cross at most
one edge. Induction therefore shows that, at step t, an observation farther
than t−1 edges from the query has zero output coefficient. Restarting a shallow
computation discards its transported intermediate state and cannot enlarge this
receptive set. A serial/topological schedule can transport more within a sweep,
but dependent local work and sequential depth must still be counted.

## 4. Risk decomposition

For any finite-second-moment predictor g(y), write

$$\theta_q-g(y)=(\theta_q-\mu_q)+(\mu_q-g(y)).$$

Condition on y and expand the square. The first term has conditional mean zero
and conditional squared mean Pi_qq. The second is constant given y, so the cross
term has conditional expectation zero. Averaging over y proves

$$\boxed{R_g=\mathbb E[(\theta_q-g(y))^2]=R_\star+\mathbb E[(\mu_q-g(y))^2],\qquad R_\star=\Pi_{qq}.}$$

For linear coefficients, C=Cov(y), k=Cov(y,theta_q), v=Sigma_qq and
w_star=C inverse times k. Consequently

$$R_t=v-2w_t^Tk+w_t^TCw_t
     =\Pi_{qq}+(w_t-w_\star)^TC(w_t-w_\star).$$

The final term is nonnegative. Correct additional computation can remove
unextracted-information error, not irreducible uncertainty in the observations.
Since the coefficient recurrence converges in finite-dimensional space, this
quadratic term tends to zero. Wrong relations generally produce a different
limit with nonzero excess risk.

**Dimension check:** every term is a squared normalized state value, with SI unit
1. Normalized MSE divides this risk by v. It is not mutual information in bits.

## 5. When exterior observations add information

Partition observations into a near subset Y_L and an exterior subset Y_E. The
residual exterior observation after predicting it from Y_L has covariance

$$S=C_{EE}-C_{EL}C_{LL}^{-1}C_{LE}$$

and covariance with the remaining query residual

$$c=k_E-C_{EL}C_{LL}^{-1}k_L.$$

Because the full observation covariance is positive definite, its Schur
complement S is positive definite. Conditioning first on Y_L and then on its
exterior residual, Gaussian square completion as in section 2 subtracts the
quadratic form

$$\boxed{R_L-R_{LE}=c^TS^{-1}c\ge0.}$$

The reduction is strictly positive exactly when c is nonzero. More exterior
nodes alone do not guarantee this condition. Redundant or conditionally
irrelevant observations need not improve the best prediction.

## 6. Exact sign accuracy

For a nonzero linear predictor, the query and prediction are jointly centered
Gaussian. Their correlation is

$$\rho=\frac{w_t^Tk}{\sqrt{v\,w_t^TCw_t}}.$$

After positive rescaling, represent them as Z_1 and
rho Z_1 + sqrt(1−rho squared) Z_2, where the two Z variables are independent
standard normals. The two-dimensional normal density depends only on radius,
so its angle is uniform. The separating normal vectors have angle
phi=arccos(rho). Sign disagreement occupies two angular wedges totaling 2phi
out of 2pi. Hence

$$\boxed{p_{\rm acc}=1-\frac{\arccos\rho}{\pi}=\frac12+\frac{\arcsin\rho}{\pi}.}$$

The limiting rho=−1 or +1 cases follow by continuity. A zero predictor has no
defined correlation; the implementation assigns the constant nonnegative
prediction, which is correct with probability one-half for the centered query.
This formula is not valid without the centered joint-Gaussian assumptions.

## 7. Counterexample interpretation and error check

The corruption experiment changes only assumed exterior relation signs and
keeps the original diagonal and observations. Absolute transition row sums are
unchanged, so convergence remains. The limit need not equal the true Bayesian
mean. Classification and squared-error objectives can disagree even while both
are measured against the same true latent values.

The gate experiment calibrates one scalar from separate corrupted training
samples. It does not reveal online ground truth, but also does not infer
per-query trustworthiness. Its cross-regime failure is part of the result.

Tests check positive definiteness, contraction, risk identities, causal zero
patterns and correspondence between coefficient-space and direct state updates.
These are numerical audits of a specified finite model. They do not establish
new algorithmic optimality, topology-only causation, floating-point interval
certification, or validity for arbitrary learned graphs and language models.
