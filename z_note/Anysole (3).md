\section{Experiments}
\label{sec:exp}
We organize evaluation around five questions: (i) How well does one checkpoint reconstruct motion under distinct inputs? (ii) When does pressure help under occlusion? (iii) Does selective fusion prevent negative transfer? (iv) Can predicted contact improve test-time multi-view fitting? (v) Can one checkpoint handle temporal changes of available sensors? The numerical results reported in this section are \emph{preliminary values transcribed from the internal experimental summary}. They require re-export from saved predictions and a final split, version and unit audit before any submission.

\subsection{Reconstruction across Input Configurations}
\label{sec:mainresults}
Table~\ref{tab:main} gives selected initial results for the unified model (configuration \texttt{V7\_Cv5}) and external reference systems. Values in the source table were reported as point estimates, without confidence intervals. Not all baselines share identical input modalities or training protocols; fair re-training and evaluation on a fixed split are necessary before making a state-of-the-art claim. In particular, the RGB-D and pressure optimization baseline cannot be treated as a direct input-matched comparator to a monocular RGB model.

\begin{table*}[t]
\centering
\caption{\textbf{Main reconstruction results across input configurations.} One \method checkpoint (\texttt{V7\_Cv5}, \texttt{ckpt\_last}) is evaluated under V-only, T-only and VT inference against external baselines matched to each input block; the three \texttt{U} rows share identical weights with no retraining or fine-tuning. \texttt{---} marks metrics unavailable for a baseline. Point estimates without confidence intervals.}
\label{tab:main}
\setlength{\tabcolsep}{2.5pt}\small
\begin{tabular}{llrrrrrrrrrr}
\toprule
Block & Method & PA$\downarrow$ & MPJPE$\downarrow$ & G-MPJPE$\downarrow$ & W-MPJPE$\downarrow$ & MPJRE$\downarrow$ & ATE$\downarrow$ & RotDrift$\downarrow$ & FootSlid$\downarrow$ & GroundPen$\downarrow$ & Jitter$\downarrow$ \\
\midrule
V & U@V & 37.50 & 53.67 & 70.92 & 87.05 & 10.87 & 41.02 & 5.11 & 9.17 & 23.40 & 704.52 \\
 & GVHMR 2024 & 50.46 & 65.46 & --- & 229.12 & 17.20 & 2271.01 & 3.84 & 32.13 & --- & 4462.11 \\
 & SAM3DB & 51.14 & 74.80 & --- & 141.01 & 15.42 & 557.54 & 4.80 & 26.76 & --- & 3901.02 \\
 & WHAM & 47.09 & 64.27 & --- & 266.77 & 13.55 & 2509.19 & 10.43 & 22.49 & --- & 2467.84 \\
\midrule
T & U@T & 89.40 & 152.49 & 185.38 & 293.46 & 20.39 & 84.84 & 33.44 & 16.43 & 42.13 & 1784.27 \\
 & Step2Motion & 81.34 & 149.84 & --- & 317.96 & --- & 1285.06 & 22.11 & 5.76 & --- & 360.55 \\
\midrule
VT & U@VT & 40.53 & 56.37 & 73.68 & 89.09 & 11.84 & 41.31 & 4.88 & 9.37 & 20.29 & 728.69 \\
 & MotionPRO & 59.09 & 77.97 & --- & 149.57 & 14.32 & 184.30 & 18.20 & 18.20 & --- & 4055.17 \\
 & VP-MoCap (FPP+PoseTransOpt) & 60.95 & 88.35 & --- & 237.91 & 13.94 & 199.19 & 5.30 & 19.17 & --- & 915.81 \\
 & MMVP pressure-toolkit & 112.24 & 225.99 & --- & 321.63 & 21.50 & 652.49 & 33.34 & 3.15 & --- & 4183.04 \\
\bottomrule
\end{tabular}
\end{table*}

The current data reveal an important trade-off. Relative to \(\unif@V\), \(\unif@VT\) has higher PA-MPJPE (40.53 versus 37.50) and MPJPE (56.37 versus 53.67) and slightly higher foot sliding (9.37 versus 9.17), while reducing ground penetration (20.29 versus 23.40). Thus, plantar evidence can improve physical plausibility without improving every pose metric. The preliminary tactile-only variant is also weaker than its specialist on some world-space and physical metrics (see supplementary comparison), so a universal \emph{no-regret} claim would be premature. This motivates the targeted fusion study instead of assuming that more modalities must yield better accuracy.

\subsection{Complementarity under Visual Occlusion}
\label{sec:occlusion}
We progressively mask lower-body image content at occlusion settings \texttt{occ000}, \texttt{occ010}, \texttt{occ020}, \texttt{occ030} and \texttt{occ050}, holding network weights constant. The initial study uses a bottom-of-person-bounding-box mask, and later experiments should replace the uniform mask with silhouette-aware and naturalistic occluders. Both \(\unif@V\) and \(\unif@VT\) are evaluated at identical occlusion levels; the tactile-only mode provides an occlusion-invariant input reference.

Table~\ref{tab:occlusion} gives \texttt{V7\_Cv5} results: PA-MPJPE 37.50/40.53 (V/VT) at \texttt{occ000}, 38.52/40.87 at \texttt{occ010}, and 40.35/42.87 at \texttt{occ020}. These numbers establish increasing visual difficulty but do not yet demonstrate an improvement in absolute pose accuracy from pressure. Subtracting each mode's own \texttt{occ000} error gives smaller \emph{occlusion-induced degradation} for VT than V at the severe levels---PA-MPJPE at \texttt{occ050} rises by $+19.55$\,mm for V versus $+16.88$\,mm for VT---and VT keeps lower ground penetration than V at every level ($54.90$ versus $59.16$\,mm at \texttt{occ050}). Because this comparison subtracts each mode's \texttt{occ000} error, it cannot establish the lowest absolute error. Subject-level uncertainty intervals and a finalized-split re-export remain pending before statistical reliability claims.

\begin{table*}[t]
\centering
\caption{\textbf{E5 occlusion study (\texttt{V7\_Cv5}, absolute errors per level).} Bottom-of-bounding-box masking at five levels with weights held fixed. VT degrades less than V at the severe levels (PA-MPJPE at \texttt{occ050}: $+19.55$ versus $+16.88$\,mm relative to each mode's \texttt{occ000}) and keeps lower ground penetration at every level, while absolute pose error stays slightly higher for VT. Point estimates without confidence intervals.}
\label{tab:occlusion}
\setlength{\tabcolsep}{2.5pt}\small
\begin{tabular}{llrrrrrrrrrr}
\toprule
Level & Input & PA$\downarrow$ & MPJPE$\downarrow$ & G-MPJPE$\downarrow$ & W-MPJPE$\downarrow$ & MPJRE$\downarrow$ & ATE$\downarrow$ & RotDrift$\downarrow$ & FootSlid$\downarrow$ & GroundPen$\downarrow$ & Jitter$\downarrow$ \\
\midrule
occ000 & V & 37.50 & 53.67 & 70.92 & 87.05 & 10.87 & 41.02 & 5.11 & 9.17 & 23.40 & 704.52 \\
 & VT & 40.53 & 56.37 & 73.68 & 89.09 & 11.84 & 41.31 & 4.88 & 9.37 & 20.29 & 728.69 \\
occ010 & V & 38.52 & 55.09 & 73.85 & 87.84 & 11.55 & 42.92 & 6.14 & 9.31 & 30.00 & 700.55 \\
 & VT & 40.87 & 57.41 & 76.30 & 90.32 & 12.18 & 42.31 & 5.82 & 9.49 & 24.71 & 732.29 \\
occ020 & V & 40.35 & 57.30 & 77.81 & 93.39 & 11.66 & 46.05 & 6.41 & 9.78 & 30.98 & 723.76 \\
 & VT & 42.87 & 59.67 & 80.15 & 95.42 & 12.40 & 45.19 & 6.16 & 9.85 & 25.28 & 749.58 \\
occ030 & V & 43.42 & 60.83 & 85.27 & 103.33 & 12.27 & 52.26 & 6.48 & 10.38 & 35.33 & 744.61 \\
 & VT & 45.12 & 63.97 & 88.10 & 104.49 & 13.06 & 51.52 & 6.24 & 10.32 & 29.04 & 772.75 \\
occ050 & V & 57.05 & 80.14 & 109.17 & 135.56 & 14.43 & 64.49 & 8.39 & 11.44 & 59.16 & 730.19 \\
 & VT & 57.41 & 82.81 & 111.21 & 134.96 & 15.38 & 63.00 & 7.37 & 11.03 & 54.90 & 776.01 \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Gated Interaction and Training Design}
\label{sec:ablation}
The ablation isolates (a) ungated fusion, (b) one shared gate, (c) separate pose/root gates, (d) enumerated three-condition training, (e) random modality dropout, and (f) full-input-only training, with all other training choices matched. Table~\ref{tab:ablate} reports the trained \texttt{V7} variant family: gating reduces U@VT PA-MPJPE from $44.55$\,mm (no gate) to $39.46$--$40.53$\,mm, enumerated three-condition training keeps U@V sharp ($37.50$ versus $43.00$\,mm under VT-only training), and the dual gate stays within $1.1$\,mm of the shared gate. Seed replication and the finalized split audit remain pending. We will additionally plot the effective gates by contact phase to test whether their values correspond to measurable improvements rather than merely varying numerically.

\begin{table}[t]
\centering
\caption{\textbf{E4: controlled fusion ablations (\texttt{V7} variant family, 740 epochs, one seed each).} U@V and U@VT PA-MPJPE (mm, \texttt{ckpt\_last}); the VT-only model is also evaluated under V to expose cross-condition generalization. Seed replication and the finalized split audit remain pending.}
\label{tab:ablate}
\setlength{\tabcolsep}{4pt}\small
\begin{tabular}{llllrr}
\toprule
Configuration & Model & Gate & Training & U@V & U@VT \\
\midrule
Unconditional fusion & V7\_Cv7 & none & enumerated & 37.01 & 44.55 \\
Single shared gate & V7\_Cv4 & shared & enumerated & 36.99 & 39.46 \\
\method (proposed) & V7\_Cv5 & dual & enumerated & 37.50 & 40.53 \\
Random modality dropout & V7\_Cv2 & dual & random & 37.35 & 40.09 \\
Always VT during training & V7\_Cv6 & dual & VT only & 43.00 & 41.35 \\
\bottomrule
\end{tabular}
\end{table}

For the motion-phase study, each foot is labeled as stance, swing, touchdown vicinity, or lift-off vicinity from synchronized contact measurements. For error \(E\), we define \(G=E(V)-E(VT)\), so positive \(G\) means that tactile fusion helps. The source document uses a conflicting sign convention in some exploratory tables; all plots and statistics must be regenerated from raw errors with this single definition. Table~\ref{tab:e6phase} reports running (session S13082): the ungated model suffers a large negative-fusion event at left stance ($G=-14.1$\,mm) and only recovers gains at swing and lift-off, while the dual gate keeps left stance positive ($G=+2.5$\,mm) and preserves the lift-off gain ($+12.2$\,mm). Across all 36 clear-view sessions (Table~\ref{tab:e6headroom}), the dual-gate model keeps VT within $3.4$\,mm of V-only, the per-frame oracle bounds the remaining headroom at $5.7$\,mm, and at moderate occlusion (\texttt{occ020}/\texttt{occ030}) VT improves ankle-foot error over V-only by $1.7$/$2.5$\,mm. Per-phase means with confidence intervals and gate calibration remain to be added.

\begin{table}[t]
\centering
\caption{\textbf{E6: running session S13082, per-phase ankle-foot errors (mm).} Cells give V error / VT error / $G$, with $G=E(V)-E(VT)$; positive $G$ means pressure helps. Phases come from synchronized contact labels with $\pm 0.1$\,s windows.}
\label{tab:e6phase}
\setlength{\tabcolsep}{6pt}\small
\begin{tabular}{llcccc}
\toprule
Model & Foot & Stance & Swing & Touchdown & Lift-off \\
\midrule
\method dual gate (V7\_Cv5) & left & 55.1/52.6/$+2.5$ & 129.6/135.0/$-5.4$ & 100.4/103.6/$-3.2$ & 149.9/137.7/$+12.2$ \\
 & right & 63.1/64.1/$-1.0$ & 105.7/107.2/$-1.5$ & 90.5/95.7/$-5.2$ & 154.1/151.3/$+2.8$ \\
No gate (V7\_Cv7) & left & 56.7/70.8/$-14.1$ & 131.3/117.4/$+13.9$ & 89.9/82.8/$+7.1$ & 157.6/143.5/$+14.1$ \\
 & right & 67.8/63.6/$+4.2$ & 105.6/113.7/$-8.1$ & 87.1/96.3/$-9.2$ & 154.6/156.4/$-1.8$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{\textbf{E6: clear-view gating headroom (ankle-foot MPJPE, mm, pelvis-aligned; 36 test sessions).} $\Delta = $ VT$-$V (negative helps). The per-frame oracle selects the better of V and VT at each frame; per-session likewise per session. Ranges cover the three T/LR/Q granularity variants and the four explicit-gain-supervision seeds.}
\label{tab:e6headroom}
\setlength{\tabcolsep}{4pt}\small
\begin{tabular}{lrrrrrr}
\toprule
Model & V & VT & $\Delta$ & Per-frame oracle & Per-session oracle & Sessions VT$<$V \\
\midrule
B (V7\_Cv5\_s1, dual gate) & 74.70 & 78.11 & $+3.41$ & 68.95 & 73.73 & 9/36 \\
V7\_Cv5TLR\_s1 (foot$\times$time gates) & 74.94 & 78.13 & $+3.19$ & 68.08 & 73.19 & 13/36 \\
T/LR/Q gate variants & 73.6--75.4 & 78.3--83.1 & $+4.3$ to $+7.7$ & --- & --- & 6--11/36 \\
Explicit-gain supervision (4 seeds) & 74.9--79.7 & 84.6--86.1 & $+6.1$ to $+9.6$ & --- & --- & 7--12/36 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Contact-Guided Multi-View Extension}
\label{sec:mvexperiments}
For \(K\in\{1,2,3,4\}\), we evaluate all available synchronized camera subsets, using a known-calibration triangulation-and-fitting reference, uncalibrated late fusion, the underlying test-time fitting baseline, \(\unif@V_K\), \(\unif@V_KT\) without the contact term, and \(\unif@V_KT\) with Eq.~\eqref{eq:contact}. An optional vision-only contact variant, if implemented, will be reported separately from insole-assisted contact. We will report the exact camera-subset variation (for \(K=4\), one full-camera subset). Primary metrics are Global-MPJPE, foot-joint error and ground penetration. To test the shared-error mechanism, we will occlude lower limbs in every available view and decompose errors into per-view residual and common displacement. \textbf{AnySole TTA rows are pending;} the calibrated reference (2--4 views) and the single-view TTA baseline are filled in Table~\ref{tab:multiview}. Results will be reported with their computation budget and any required camera calibration assumptions.

\begin{table}[t]
\centering
\caption{\textbf{E3: multi-view study.} Cells give MPJPE / Global-MPJPE (mean $\pm$ std over camera subsets; the single 4-view subset has no std). PA-MPJPE is available in the exported table.}
\label{tab:multiview}
\setlength{\tabcolsep}{3pt}\small
\begin{tabular}{lcccc}
\toprule
Method & 1 view & 2 views & 3 views & 4 views \\
\midrule
\shortstack[l]{Calibrated triangulation\\ + fitting (EasyMocap)} & \todoresult &
\shortstack{$68.94 \pm 12.80$\\ $104.29 \pm 42.73$} &
\shortstack{$51.96 \pm 4.05$\\ $60.79 \pm 8.72$} &
\shortstack{$48.47$\\ $52.60$} \\
\shortstack[l]{Test-time fitting\\ baseline (HMR 2.0)} &
\shortstack{$106.99 \pm 8.94$\\ $101.38 \pm 8.18$} & \todoresult & \todoresult & \todoresult \\
\shortstack[l]{\method TTA,\\ no contact} & \todoresult & \todoresult & \todoresult & \todoresult \\
\shortstack[l]{\method +\\ contact constraint} & \todoresult & \todoresult & \todoresult & \todoresult \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Dynamic Inputs and Real Captures}
\label{sec:dynamic}
For E7, we apply an identical sequence of availability masks to a continuous ground-truth motion, including transitions among tactile-only, monocular, joint visuo-tactile, and multi-view input. A single \method checkpoint is compared with a specialist ensemble that switches models at each configuration transition. We measure overall reconstruction error, transition-local discontinuity and acceleration jitter within a fixed time neighborhood, and recovery after vision returns. All systems receive identical observed inputs and are evaluated after consistent trajectory alignment; simply keeping one checkpoint is not sufficient to guarantee smooth motion across mask changes. The current implementation switches availability at temporal-window boundaries.

\begin{table}[t]
\centering
\caption{\textbf{E7: one checkpoint versus specialist ensemble on representative motions (ankle-foot MPJPE, mm, pelvis-aligned).} Stages switch at window boundaries: V $\to$ T $\to$ VT $\to$ Vi $\to$ ViT. Ensemble: V=WHAM, T=Step2Motion, VT=MotionPRO, Vi/ViT=EasyMocap (4 views). \method Vi/ViT rows remain pending.}
\label{tab:dynamic}
\setlength{\tabcolsep}{6pt}\small
\begin{tabular}{llrrrrr}
\toprule
Motion & System & V & T & VT & Vi & ViT \\
\midrule
Squat (6 sessions) & \method (V7\_Cv5) & 50.2 & 294.8 & 62.6 & --- & --- \\
 & Ensemble & 83.6 & 354.3 & 68.1 & 70.5 & 48.0 \\
Sit-stand (5 sessions) & \method (V7\_Cv5) & 50.2 & 170.3 & 56.6 & --- & --- \\
 & Ensemble & 88.5 & 404.3 & 59.1 & 86.0 & 67.0 \\
Lateral shift (5 sessions) & \method (V7\_Cv5) & 95.9 & 245.2 & 85.9 & --- & --- \\
 & Ensemble & 121.7 & 242.3 & 134.1 & 66.9 & 50.1 \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:dynamic} reports the three motions with the largest session counts (squat 6, sit-stand 5, lateral 5; the remaining six per-action tables are exported under \texttt{results\_display/Etest/E7\_*}). The single checkpoint wins every V and VT stage shown, wins the T stage for squat and sit-stand, and stays within $2.9$\,mm of the T specialist for lateral; its Vi/ViT rows remain pending. The continuous real-capture teaser, timing protocol and baseline transitions remain pending.

\subsection{Complementary Information Probe}
We fit a frozen-feature linear probe comparing visual memory \(\mathbf M_V\), tactile memory \(\mathbf M_T\), and concatenated \([\mathbf M_V,\mathbf M_T]\). All use the same ridge-regression protocol to predict pelvis-relative ankle/foot targets; regularization is selected on validation subjects. Table~\ref{tab:probe} reports test errors on seven targets. The concatenated representation beats \(\mathbf M_V\) on the two vertical components (position-y $60.14$ versus $68.89$\,mm; stance-y $60.15$ versus $68.67$\,mm), so the extra linearly decodable tactile information concentrates in the vertical component, without claiming that the downstream motion decoder necessarily exploits it. The original position criterion (joint better than both memories) held on validation but did not replicate on test. These E8 measurements will appear in the supplementary material.

\begin{table}[t]
\centering
\caption{\textbf{E8: frozen-feature ridge probe (\texttt{V7\_Cv5}).} Predicting pelvis-relative ankle/foot targets from visual memory $\mathbf M_V$, tactile memory $\mathbf M_T$, and their concatenation; $\lambda$ selected on validation. Errors in mm (velocity in mm/s; contact accuracy in \%, higher is better).}
\label{tab:probe}
\setlength{\tabcolsep}{5pt}\small
\begin{tabular}{lrrrrrrr}
\toprule
Probe input & Position & Position-y & Stance-y & Velocity & Global & Ground & Contact $\uparrow$ \\
\midrule
$\mathbf M_V$ & 116.78 & 68.89 & 68.67 & 427.56 & 348.50 & 33.93 & 90.12 \\
$\mathbf M_T$ & 204.46 & 116.75 & 116.60 & 496.58 & 530.24 & 50.54 & 84.98 \\
$[\mathbf M_V, \mathbf M_T]$ & 122.05 & 60.14 & 60.15 & 459.60 & 409.87 & 43.93 & 85.25 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Internal Data Provenance}
\label{sec:suppdata}
The tables above transcribe the exported files under \texttt{results\_display/Etest}: E1 experts \texttt{E1\_CleanPerformance/Full\_Table1\_E1\_Experts.csv} (variant family \texttt{V7\_Cv1}--\texttt{Cv7}; the \texttt{V7\_Cv5} rows appear in Table~\ref{tab:main}); E2 \texttt{E2\_MainPerformance/Full\_Table1--4\_E2\_*}; E3 \texttt{E3\_MultiViewConsistency/Full\_Table1\_E3\_MultiView.csv} (includes PA-MPJPE); E4 \texttt{E4\_ComponentAblation/Full\_Table1\_E4\_Ablation\_AllMetrics.csv} (all ten metrics plus $\Delta$(VT$-$V) rows); E5 \texttt{E5\_OcclusionRobustness/Full\_Table1\_E5\_Occlusion.csv} (includes $\Delta$ rows); E6 per-action phase tables and paired statistics under \texttt{E6\_ContactPhaseAnalysis/} (\texttt{E6\_接触相位分析.md}, \texttt{e6\_paired\_stats\_*.json}); E7 all nine motions \texttt{E7\_DynamicModalityInputs/Full\_Table1\_E7\_DynamicStages.csv}; E8 \texttt{E8\_ComplementarityProbe/Full\_Table1\_E8\_Probe.csv} (test/val/$\lambda$ per target).
