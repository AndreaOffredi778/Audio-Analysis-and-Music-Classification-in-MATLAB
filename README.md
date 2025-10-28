Code Description
The code performs analysis on audio tracks divided into three classes:
    • Disco
    • Metal
    • HipHop
Each class is organized into two subfolders containing 6 audio files, split into training and test sets as follows:
    • 50% of files for training (3 files)
    • 50% of files for testing (3 files)
Folder Structure
    • IAM: The main file containing all operations required to address the tasks and generate the requested values or plots.
    • function: Contains utility functions for audio feature extraction, such as MFCC (Mel Frequency Cepstral Coefficients) and CHROMA (chroma features).
Main functions include:
    • extract_from_path and extract_from_path_chroma: extract MFCC and CHROMA features.
    • sigmerge, specsub: handle noise addition and spectral subtraction.
    • add_noise_and_denoise: adds noise to the test set.
    • code: Contains all files necessary to compute, visualize, and extract the values or plots referenced in IAM.
Code Workflow
File Reading:
The code reads paths for training and test files across all three audio classes.
Feature Extraction:
Using the specific functions, features (MFCC and CHROMA) are extracted for both training and test sets. Sets are normalized to ensure consistent dimensions.
Combining Feature Sets:
The extracted feature sets (MFCC and CHROMA) are concatenated into a single matrix called ALL, required for combined feature analysis. Features are identified as follows:
    • CHROMA: type 1
    • MFCC: type 2
    • ALL (MFCC + CHROMA): type 3
Training with k-NN (k-Nearest Neighbors):
The k-NN algorithm classifies the test set based on the training set. Recognition rates are calculated for each feature type (MFCC, CHROMA, ALL), and corresponding plots and matrices are generated. Comparisons with Decision Tree (DT) performance are also included.
Training with Decision Tree (DT):
Using fitcree and predict, a decision tree is generated from the training set and labels. Matrices are also created to compare with k-NN results.
Searching for Maximum Recognition Rate:
For k-NN, the maximum recognition rate across different k-values is found; for DT, the maximum rate from the matrices is identified. Summary rates are printed for each k.
Adding Noise:
The add_noise_and_denoise function concatenates babble.wav to all test files for each class. Parameters include file paths, noise filename, SNR (set to 5), and feature type. Noisy files are saved in the noisyfile directory with subfolders per class. Spectrograms for noisy files are generated with three plots per class.
Performance on Noisy Test Set:
After adding noise, modified files are used to evaluate k-NN and DT performance. Features are re-extracted from noisy files, and corresponding matrices are generated for each feature type.

Results from IAM.m Execution:

extract train features for CHROMA...
extract test features for CHROMA...
extract train features for MFCC...
extract test features for MFCC...
link of MFCC + CHROMA

Set-up the kNN... number of neighbors: 1
Recognition Rate with 1 Neighbors: 41.1015361558636
Set-up the kNN... number of neighbors: 10
Recognition Rate with 10 Neighbors: 43.9865118021731
Set-up the kNN... number of neighbors: 20
Recognition Rate with 20 Neighbors: 44.8107905582615

Recap CHROMA:
1 -> 41.1015361558636
10 -> 43.9865118021731
20 -> 44.8107905582615

Best-performing k: 20
Maximum recognition rate: 44.8108%

Set-up the kNN... number of neighbors: 1
Recognition Rate with 1 Neighbors: 49.7564630947921
Set-up the kNN... number of neighbors: 10
Recognition Rate with 10 Neighbors: 53.8403896590483
Set-up the kNN... number of neighbors: 20
Recognition Rate with 20 Neighbors: 55.8636193330835

Recap MFCC:
1 -> 49.7564630947921
10 -> 53.8403896590483
20 -> 55.8636193330835
Best-performing k: 20
Maximum recognition rate: 55.8636%

Set-up the kNN... number of neighbors: 1
Recognition Rate with 1 Neighbors: 51.7047583364556
Set-up the kNN... number of neighbors: 10
Recognition Rate with 10 Neighbors: 54.7770700636943
Set-up the kNN... number of neighbors: 20
Recognition Rate with 20 Neighbors: 55.9010865492694

Recap CHROMA + MFCC:
1 -> 51.7047583364556
10 -> 54.7770700636943
20 -> 55.9010865492694
Best-performing k: 20
Maximum recognition rate: 55.9011%

CHROMA rates
rate_knnCHROMA = 44.8108
rate_DTCHROMA = 40.8393

MFCC rates
rate_knnMFCC = 55.8636
rate_DTMFCC = 43.1622

MFCC + CHROMA rates
rate_knnALL = 55.9011
rate_DTALL =42.9374

Searching for optimum k-NN rate
k-NN Maximum Recognition Rate is: 55.9010865492694
and it is achieved with MFCC + CHROMA 20 Neighbors.

Searching for optimum DT rate
DT Maximum Rate is: 43.1622330460847
and it is achieved with MFCC 

Adding noise babble.wav to file disco.00001.wav at SNR 5...
Saving the noisy file...
Denoising...
……………….
Processing complete!
plotting the spectrograms...
plotting the spectrograms...
plotting the spectrograms…

extraction features of noisy train set
link of MFCC + CHROMA

Set-up the kNN... number of neighbors: 1
Recognition Rate with 1 Neighbors: 37.5421506182091
Set-up the kNN... number of neighbors: 10
Recognition Rate with 10 Neighbors: 40.3521918321469
Set-up the kNN... number of neighbors: 20
Recognition Rate with 20 Neighbors: 41.2139378044211

Recap CHROMA:
1 -> 37.5421506182091
10 -> 40.3521918321469
20 -> 41.2139378044211
Best-performing k: 20
Maximum recognition rate: 41.2139%

Set-up the kNN... number of neighbors: 1
Recognition Rate with 1 Neighbors: 50.1686024728363
Set-up the kNN... number of neighbors: 10
Recognition Rate with 10 Neighbors: 53.3907830648183
Set-up the kNN... number of neighbors: 20
Recognition Rate with 20 Neighbors: 53.8403896590483

Recap MFCC:
1 -> 50.1686024728363
10 -> 53.3907830648183
20 -> 53.8403896590483
Best-performing k: 20
Maximum recognition rate: 53.8404%

Set-up the kNN... number of neighbors: 1
Recognition Rate with 1 Neighbors: 52.6414387411015
Set-up the kNN... number of neighbors: 10
Recognition Rate with 10 Neighbors: 55.1142750093668
Set-up the kNN... number of neighbors: 20
Recognition Rate with 20 Neighbors: 56.4256275758711

Recap CHROMA + MFCC:
1 -> 52.6414387411015
10 -> 55.1142750093668
20 -> 56.4256275758711
Best-performing k: 20
Maximum recognition rate: 56.4256%
<img width="682" height="510" alt="image" src="https://github.com/user-attachments/assets/eb269aeb-3ae8-4928-b6ab-c32e6c7acb84" />
<img width="674" height="510" alt="image" src="https://github.com/user-attachments/assets/3cd32b88-cc62-4b50-b488-ff1919269b74" />
<img width="681" height="506" alt="image" src="https://github.com/user-attachments/assets/5034beb2-726f-434a-8f80-8fae0470be42" />
<img width="682" height="510" alt="image" src="https://github.com/user-attachments/assets/2857565d-836c-4f7e-9adf-415587551a71" />
<img width="673" height="514" alt="image" src="https://github.com/user-attachments/assets/cfb3677f-9683-4cf7-8ec3-e9b5538b664e" />
<img width="681" height="514" alt="image" src="https://github.com/user-attachments/assets/5020fc51-81da-4d55-9096-734bb4cee7b9" />
<img width="687" height="516" alt="image" src="https://github.com/user-attachments/assets/5810c747-f46f-4cae-96bc-5fda1a0ce109" />

