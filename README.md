##EEG Data Analysis and Classification
Overview
This project involves generating synthetic EEG data, extracting features, and applying a Probabilistic Neural Network (PNN) to classify EEG segments. The primary focus is on distinguishing between different conditions, such as normal, interictal, and ictal states, as well as localizing epileptogenic zones.

The pipeline includes:

Generating Synthetic EEG Data: Simulate EEG data for different conditions.
Feature Extraction: Extract relevant features from the EEG data.
Preprocessing: Normalize the features.
Classification: Apply a Probabilistic Neural Network (PNN) for classification.
Evaluation: Assess model performance using cross-validation and various metrics.
Getting Started

###Prerequisites
Python 3.x
NumPy
Pandas
SciPy
scikit-learn
You can install the required packages using pip:

pip install numpy pandas scipy scikit-learn

##Generating Synthetic EEG Data
The synthetic EEG data is generated based on three conditions:

Normal: Standard Gaussian noise.
Interictal: Gaussian noise with added sinusoidal oscillations.
Ictal: Higher amplitude Gaussian noise with more pronounced oscillations.
Feature Extraction
Features are extracted using various methods:

Power Spectral Features: Power in different frequency bands.
Fractal Dimensions: Petrosian and Higuchi fractal dimensions.
Hjorth Parameters: Activity, mobility, and complexity.
Statistical Features: Mean, standard deviation, and their absolute values.
Classification
A Probabilistic Neural Network (PNN) is implemented to classify EEG segments. The model is evaluated using 10-fold cross-validation, and performance metrics include accuracy, precision, recall, and F1 score.

Running the Code
To run the experiments and evaluate the model, execute the main.py script:

python main.py


##Results
The script outputs the results of various classification experiments:

Normal vs Interictal
Normal vs Ictal
Interictal vs Ictal
Epileptogenic zone vs Opposite hemisphere
It displays accuracy, precision, recall, F1 score, and average classification time.

##Example Output

Experiment Results:
------------------
Normal vs Interictal:
  Accuracy: 99.500
  Precision: 99.500
  Recall: 99.500
  F1 Score: 99.500
  Avg. Classification Time: 0.01 seconds

Normal vs Ictal:
  Accuracy: 98.300
  Precision: 98.300
  Recall: 98.300
  F1 Score: 98.300
  Avg. Classification Time: 0.01 seconds

Interictal vs Ictal:
  Accuracy: 96.700
  Precision: 96.700
  Recall: 96.700
  F1 Score: 96.700
  Avg. Classification Time: 0.01 seconds

Epileptogenic zone vs Opposite hemisphere:
  Accuracy: 77.500
  Precision: 77.500
  Recall: 77.500
  F1 Score: 77.500
  Avg. Classification Time: 0.01 seconds


Notes
The current implementation uses synthetic data. For real applications, replace the data generation step with actual EEG data acquisition.
The choice of PNN parameters (e.g., spread) can be tuned for better performance.
Contributing
Feel free to submit issues or pull requests if you have suggestions or improvements. Contributions are welcome!

License
This project is licensed under the MIT License. See the LICENSE file for details