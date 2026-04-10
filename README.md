# QML-2026-Deloitte
We present a hybrid Quantum Machine Learning (QML) framework for wildfire risk
modeling across 2,593 California ZIP codes using historical climate and fire data
spanning 2018--2022, addressing Task~1 (fire risk prediction for 2023) and
Task~2 (insurance premium forecasting). Wildfire prediction is formulated as a
zero-inflated count regression problem: predicting the expected number of fire
events per ZIP code per month.

For Task~1A, we train and evaluate three Variational Quantum Circuit (VQC) architectures
implemented in Qiskit with the Qiskit Machine Learning connector to PyTorch.
All circuits use 9~qubits with angle encoding across nine engineered features
capturing temperature, precipitation, drought accumulation, seasonality, and geography.
The models differ in circuit depth, entanglement topology, and output head: \textbf{qml\_3}
is a single-circuit binary classifier (reps~=~3, all-to-all entanglement, 54 trainable
parameters); \textbf{qml\_5} is a two-stage binary-then-count regressor (reps~=~1,
ring entanglement, trained on a natural-distribution sample); and \textbf{qml\_6}
uses the same two-stage architecture as qml\_5 but trained on a class-balanced
sample (30\% fire fraction). Annual ZIP-level risk scores are produced by summing
monthly predicted fire counts across all 12 months of 2023.

For Task~1B, we benchmark all three QML models against five classical models
(Dummy, Ridge Regression, Random Forest, Gradient Boosting, XGBoost) using a
strict temporal split (train: 2018--2020; validate: 2021; test: 2022).
The best classical model, Gradient Boosting trained on 93,348 rows, achieves
AUC-ROC~=~0.873 on the 2022 test set. The best QML model (qml\_3) achieves
AUC-ROC~=~0.667, trained on only 1,800 rows due to quantum simulator constraints.
We demonstrate that the performance gap is attributable primarily to the 52$\times$
training-data disparity rather than circuit expressivity, and discuss how
fault-tolerant quantum hardware would close this gap.