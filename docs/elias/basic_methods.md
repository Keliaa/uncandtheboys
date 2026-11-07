First, distinguish three tasks:

- **Fault detection:** Find signs of a problem.
- **Fault diagnosis:** Identify the likely cause.
- **Failure prediction:** Estimate future failure risk or time to failure.

The project already includes four working **baselines**: reference solutions for comparing detector performance.

| Baseline | How it works | Role in comparison |
|---|---|---|
| **B0: Existing alarms** | Uses the supplied fixed-limit alarm records. | Shows what the existing alarm system achieves. |
| **B1: Mode-aware thresholds** | Detects sustained deviations from normal measurements at similar pump speeds. | Shows what simple statistical rules can achieve. |
| **B2: CUSUM** | Accumulates deviations to detect persistent changes. | Tests a different statistical approach to developing problems. |
| **B3: Isolation forest** | Learns normal patterns and detects unusual sensor combinations. | Provides a machine-learning comparison. |

There are four because different methods can detect different problems and produce different numbers of false warnings.
They are alternative detectors. The same scorer compares their detection, timing, diagnosis, and false warnings.
The [baseline runner](../../../baselines/run_baselines.py) evaluates all four. [Quickstart](../../../examples/quickstart.py) runs B1 on the starter data.

The supplied data support fault detection, diagnosis, and forecasts of sensor behaviour.
Exact remaining-life prediction needs additional lifetime data. The alarm deadline is not a breakdown time.
This list covers plausible methods and their uses. A feature is a calculated input. A residual is measured value minus predicted value.

Statistical methods describe normal operation and detect changes.

| Method | Meaning and use with these sensors |
|---|---|
| **Descriptive statistics** | Mean and median describe usual values. Standard deviation and percentiles describe variation. They form sensor features. |
| **Thresholds and robust z-scores** | Measure distance from normal values, optionally for each speed or operating mode. B1 uses medians and median absolute deviations. |
| **Moving average and exponential smoothing** | Reduce short fluctuations to reveal slower changes in temperature, pressure, or vibration. |
| **Control charts: Shewhart, CUSUM, EWMA** | Detect large deviations or small persistent changes. CUSUM, used by B2, accumulates deviations. EWMA gives recent data more weight. |
| **Change-point detection** | Find a change in signal level, variation, or trend. Useful for calibration shifts and changes in machine condition. |
| **Statistical tests: t-test, ANOVA, distribution tests** | Compare operating periods or groups. Closely spaced samples are related, so ordinary independence assumptions need care. |
| **Correlation and cross-correlation** | Measure relationships between sensors, including time delays. Useful for checking whether related signals change together. Correlation does not prove cause. |

Signal-processing methods use vibration recordings or suitable time sequences.

| Method | Meaning and use with these sensors |
|---|---|
| **RMS, crest factor, kurtosis** | Describe signal size and large peaks. Useful for vibration and impacts. RMS and crest factor are already supplied. |
| **FFT and power spectral density** | Show signal frequencies and their strength. Useful for imbalance, bearing damage, and repeated mechanical disturbances. |
| **Envelope analysis** | Extract changes in vibration amplitude to reveal repeated bearing impacts. |
| **Order analysis** | Relate vibration frequency to shaft rotation frequency. Useful when pump speed changes. |
| **Short-time Fourier transform and wavelets** | Show how frequency content changes during a recording. Useful for short impacts and changing signals. |

High-frequency analysis needs the raw vibration recordings. One-minute summaries do not retain those details.

These methods find unusual measurements or operating groups without a fault label for every example.

| Method | Meaning and use with these sensors |
|---|---|
| **Principal component analysis (PCA)** | Combine related sensors into fewer components. Unusual component values or reconstruction errors can reveal changes. |
| **Mahalanobis distance and Hotelling's T-squared** | Measure departure from a normal sensor combination while accounting for correlations. Useful for monitoring several sensors together. |
| **Gaussian mixture model (GMM)** | Describe normal data as several probability distributions. Useful for multiple operating modes and low-likelihood anomaly detection. |
| **K-means** | Divide similar measurements into groups. Groups can represent operating modes but need interpretation. |
| **DBSCAN** | Find dense groups and isolated measurements. Useful when operating groups have irregular shapes. |
| **Local outlier factor (LOF)** | Compare a measurement's local density with its neighbours. Useful for unusual points within different operating regions. |
| **Isolation forest** | Find measurements that are easy to separate from other data. B3 uses this for anomaly detection. |
| **One-class SVM** | Learn a boundary around normal measurements. Values outside the boundary are candidates for investigation. |

These are established [anomaly-detection approaches](https://scikit-learn.org/stable/modules/outlier_detection.html). An anomaly score does not identify a fault by itself.

Supervised methods use known targets, such as fault labels or measured sensor values.

| Method | Meaning and use with these sensors |
|---|---|
| **Linear or regularised regression** | Predict a sensor value from speed, load, and other sensors. Unexpected residuals can indicate faults. Regularisation limits model complexity. |
| **Logistic regression** | Estimate class probabilities, such as normal versus faulty or machine versus sensor problem. |
| **Decision tree** | Learn decision rules from features. Useful for inspecting how fault classifications are made. |
| **Random forest** | Combine many decision trees for fault classification or expected-value prediction. |
| **Gradient-boosted trees** | Add trees that correct previous errors. XGBoost and LightGBM are examples for sensor-feature tables. |
| **Support vector machine (SVM)** | Learn boundaries between known operating or fault classes. |
| **K-nearest neighbours (KNN)** | Compare measurements with similar labelled examples to classify faults or estimate expected values. |
| **Gaussian naive Bayes** | Use sensor-feature distributions to classify faults. It assumes features are independent within each class, which may be inaccurate for related sensors. |

The [fault-diagnosis catalogue](https://www.mathworks.com/help/predmaint/detect-and-diagnose-faults.html) includes standard classifiers and dynamic models for sensor data.

Probabilistic and dynamic models combine evidence, time, and physical relationships.

| Method | Meaning and use with these sensors |
|---|---|
| **Bayesian inference and Bayesian networks** | Combine prior fault probabilities with sensor evidence. Networks can represent dependencies between faults and symptoms. Requires justified priors and an evidence model. |
| **Gaussian process regression** | Predict expected sensor values with uncertainty under the model. Useful for residual monitoring with a manageable training set. |
| **AR, ARIMA, ARX models** | Predict sensor values from their history. ARX also uses other inputs, such as pump speed. Forecast errors can reveal faults. |
| **Kalman filter and state-space models** | Combine a dynamic model with noisy measurements to estimate hidden state. Prediction residuals can reveal faults. |
| **Particle filter** | Track possible states and their probabilities for nonlinear models. Useful when a simple Kalman model is insufficient. |
| **Hidden Markov model (HMM)** | Estimate hidden operating or health states and transitions from sensor sequences. Requires a suitable state and transition model. |
| **Physics-based residuals and parameter estimation** | Compare measurements with pump curves or a fitted physical model. Changes in residuals or estimated parameters can indicate wear or restriction. |
| **Rule-based sensor fusion** | Combine evidence across sensors into a likely cause. An isolated sensor change can suggest an instrument problem. |

**Bayesian diagnosis is a realistic approach here.** Gaussian naive Bayes is one simple classifier; Bayesian networks can describe more specific fault relationships.
An example is updating the probability of inlet blockage after low suction pressure and rising pump speed.
The result depends on the evidence model. A probability output is not automatically a well-calibrated confidence estimate.
See [naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) and [Gaussian process regression](https://scikit-learn.org/stable/modules/gaussian_process.html).

Neural networks are also plausible, with suitable inputs and model size.

| Method | Meaning and use with these sensors |
|---|---|
| **Multilayer perceptron (MLP)** | A basic neural network for classifying sensor-feature rows or predicting expected values. |
| **Autoencoder** | Learn to reproduce normal features or sequences. Large reproduction errors can indicate unusual behaviour. |
| **1D convolutional neural network (CNN)** | Learn local patterns in vibration recordings or sensor sequences. |
| **GRU and LSTM recurrent networks** | Learn relationships across a sequence of measurements. Useful for fault classification or sensor forecasting. |
| **Temporal convolutional network (TCN)** | Use convolutions across a time window to learn sequence patterns. A causal version uses past measurements only. |

The training data contain seven pumps and 27 labelled fault episodes. Adjacent rows are not independent fault examples.
Network size and the number of fault examples limit what can be learned reliably. Model accuracy must be measured on different pumps.

Learning terms describe how these methods use data:

| Term | Meaning |
|---|---|
| **Supervised learning** | Learn from measurements paired with known targets. |
| **Unsupervised learning** | Find patterns without an answer for each example. |
| **Self-supervised learning** | Create learning targets from the measurements, such as reconstruction or prediction of hidden values. |
| **Online learning** | Update a model as data arrive. Requires an explicit update method. |

B3 learns normal patterns after excluding labelled training faults. It does not continuously retrain.
Online updates can absorb a developing fault into normal behaviour. State estimation, such as Kalman filtering, does not necessarily update model parameters.
Dataset fields, splits, and cost analysis are described in [data_summary.md](data_summary.md).
