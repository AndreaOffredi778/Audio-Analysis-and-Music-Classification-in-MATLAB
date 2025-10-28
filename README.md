# Code Description

The code performs analysis on audio tracks divided into three classes:

- **Disco**
- **Metal**
- **HipHop**

Each class is organized into two subfolders containing 6 audio files, split into training and test sets as follows:

- 50% of files for training (3 files)  
- 50% of files for testing (3 files)  

---

## Folder Structure

- **IAM:** The main file containing all operations required to address the tasks and generate the requested values or plots.  
- **function:** Contains utility functions for audio feature extraction, such as **MFCC** (Mel Frequency Cepstral Coefficients) and **CHROMA** (chroma features).  

### Main Functions

- `extract_from_path` and `extract_from_path_chroma`: extract MFCC and CHROMA features  
- `sigmerge`, `specsub`: handle noise addition and spectral subtraction  
- `add_noise_and_denoise`: adds noise to the test set  
- **code:** Contains all files necessary to compute, visualize, and extract the values or plots referenced in IAM  

---

## Code Workflow

### 1. File Reading
The code reads paths for training and test files across all three audio classes.

### 2. Feature Extraction
Using the specific functions, features (**MFCC** and **CHROMA**) are extracted for both training and test sets.  
Sets are normalized to ensure consistent dimensions.

### 3. Combining Feature Sets
The extracted feature sets (**MFCC** and **CHROMA**) are concatenated into a single matrix called `ALL`, required for combined feature analysis.  
Feature types:

- **CHROMA:** type 1  
- **MFCC:** type 2  
- **ALL (MFCC + CHROMA):** type 3  

### 4. Training with k-NN (k-Nearest Neighbors)
The k-NN algorithm classifies the test set based on the training set.  
Recognition rates are calculated for each feature type (**MFCC**, **CHROMA**, **ALL**), and corresponding plots and matrices are generated.  
Comparisons with **Decision Tree (DT)** performance are also included.

### 5. Training with Decision Tree (DT)
Using `fitcree` and `predict`, a decision tree is generated from the training set and labels.  
Matrices are also created to compare with k-NN results.

### 6. Searching for Maximum Recognition Rate
- For k-NN: the maximum recognition rate across different k-values is found.  
- For DT: the maximum rate from the matrices is identified.  
- Summary rates are printed for each k.

### 7. Adding Noise
The `add_noise_and_denoise` function concatenates `babble.wav` to all test files for each class.  

Parameters:

- File paths  
- Noise filename (`babble.wav`)  
- SNR (set to 5)  
- Feature type  

Noisy files are saved in the `noisyfile` directory with subfolders per class.  
Spectrograms for noisy files are generated, with three plots per class.

### 8. Performance on Noisy Test Set
After adding noise:

- Modified files are used to evaluate k-NN and DT performance.  
- Features are re-extracted from noisy files.  
- Corresponding matrices are generated for each feature type.  

## Recognition Rates and Noise Processing Results

### MFCC and MFCC + CHROMA Rates

- **MFCC**
  - k-NN: 55.8636%
  - Decision Tree (DT): 43.1622%

- **MFCC + CHROMA**
  - k-NN: 55.9011%
  - DT: 42.9374%

**Optimum Recognition Rates**

- k-NN Maximum: 55.9011% with MFCC + CHROMA using 20 neighbors  
- DT Maximum: 43.1622% with MFCC

---

### Noisy Audio Processing (`babble.wav`, SNR = 5)

- Adding noise to file `disco.00001.wav`  
- Saving noisy files and denoising  
- Plotting spectrograms for each class  

**Feature extraction on noisy train set:** MFCC + CHROMA  

---

### k-NN Performance on Noisy Set

**CHROMA Features:**

| k | Recognition Rate (%) |
|---|--------------------|
| 1 | 37.54              |
| 10| 40.35              |
| 20| 41.21              |

- **Best-performing k:** 20  
- **Maximum recognition rate:** 41.21%

**MFCC Features:**

| k | Recognition Rate (%) |
|---|--------------------|
| 1 | 50.17              |
| 10| 53.39              |
| 20| 53.84              |

- **Best-performing k:** 20  
- **Maximum recognition rate:** 53.84%

**CHROMA + MFCC Features:**

| k | Recognition Rate (%) |
|---|--------------------|
| 1 | 52.64              |
| 10| 55.11              |
| 20| 56.43              |

- **Best-performing k:** 20  
- **Maximum recognition rate:** 56.43%

<img width="682" height="510" alt="image" src="https://github.com/user-attachments/assets/eb269aeb-3ae8-4928-b6ab-c32e6c7acb84" />
<img width="674" height="510" alt="image" src="https://github.com/user-attachments/assets/3cd32b88-cc62-4b50-b488-ff1919269b74" />
<img width="681" height="506" alt="image" src="https://github.com/user-attachments/assets/5034beb2-726f-434a-8f80-8fae0470be42" />
<img width="682" height="510" alt="image" src="https://github.com/user-attachments/assets/2857565d-836c-4f7e-9adf-415587551a71" />
<img width="673" height="514" alt="image" src="https://github.com/user-attachments/assets/cfb3677f-9683-4cf7-8ec3-e9b5538b664e" />
<img width="681" height="514" alt="image" src="https://github.com/user-attachments/assets/5020fc51-81da-4d55-9096-734bb4cee7b9" />
<img width="687" height="516" alt="image" src="https://github.com/user-attachments/assets/5810c747-f46f-4cae-96bc-5fda1a0ce109" />

