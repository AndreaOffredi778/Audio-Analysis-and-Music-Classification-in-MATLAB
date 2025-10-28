clear; clc
addpath(genpath(pwd)) % aggiunta delle cartelle e sottocartelle
windowLength = 0.5;   % Lunghezza della finestra (in secondi)
stepLength = 0.1;     % Passo tra finestre adiacenti (in secondi)

% Percorsi delle cartelle contenenti i dati di training e test
traindiscoPath = [pwd,'\disco\train\'];
trainmetalPath = [pwd,'\metal\train\'];
trainhiphopPath = [pwd,'\hiphop\train\'];
testdiscoPath = [pwd,'\disco\test\'];
testmetalPath = [pwd,'\metal\test\'];
testhiphopPath = [pwd,'\hiphop\test\'];

% Estrazione delle feature CHROMA dai dati di training
disp('extract train features for CHROMA...')
train_disco_features = extract_from_path_chroma(traindiscoPath,'wav',windowLength,stepLength);
train_metal_features = extract_from_path_chroma(trainmetalPath,'wav',windowLength,stepLength);
train_hiphop_features = extract_from_path_chroma(trainhiphopPath,'wav',windowLength,stepLength);

% Etichettatura dei dati di training (1=disco, 2=metal, 3=hiphop)
labeldisco = repmat(1,length(train_disco_features),1);
labelmetal = repmat(2,length(train_metal_features),1);
labelhiphop = repmat(3,length(train_hiphop_features),1);

% Concatenazione delle feature e delle etichette
all_trainCHROMA = [train_disco_features train_metal_features train_hiphop_features];
all_labelsCHROMA = [labeldisco; labelmetal; labelhiphop];

% Normalizzazione delle feature CHROMA
all_trainCHROMA = all_trainCHROMA';
mn = mean(all_trainCHROMA);
st = std(all_trainCHROMA);
all_trainCHROMA =  (all_trainCHROMA - repmat(mn,size(all_trainCHROMA,1),1))./repmat(st,size(all_trainCHROMA,1),1);

% Estrazione delle feature CHROMA dai dati di test
disp('extract test features for CHROMA...')
test_disco_features = extract_from_path_chroma(testdiscoPath,'wav',windowLength,stepLength);
test_metal_features = extract_from_path_chroma(testmetalPath,'wav',windowLength,stepLength);
test_hiphop_features = extract_from_path_chroma(testhiphopPath,'wav',windowLength,stepLength);

% Concatenazione delle feature e delle etichette di test
all_testCHROMA = [test_disco_features test_metal_features test_hiphop_features];
testdiscoLabel = repmat(1,length(test_disco_features),1);
testmetalLabel = repmat(2,length(test_metal_features),1);
testhiphopLabel = repmat(3,length(test_hiphop_features),1);
ground_truthCHROMA = [testdiscoLabel; testmetalLabel; testhiphopLabel];

% Normalizzazione dei dati di test CHROMA
all_testCHROMA = all_testCHROMA';
all_testCHROMA =  (all_testCHROMA - repmat(mn,size(all_testCHROMA,1),1))./repmat(st,size(all_testCHROMA,1),1);

% Estrazione delle feature MFCC dai dati di training
disp('extract train features for MFCC...')
train_disco_features = extract_from_path(traindiscoPath,'wav',windowLength,stepLength);
train_metal_features = extract_from_path(trainmetalPath,'wav',windowLength,stepLength);
train_hiphop_features = extract_from_path(trainhiphopPath,'wav',windowLength,stepLength);

% Etichettatura dei dati di training (1=disco, 2=metal, 3=hiphop)
labeldisco = repmat(1,length(train_disco_features),1);
labelmetal = repmat(2,length(train_metal_features),1);
labelhiphop = repmat(3,length(train_hiphop_features),1);

% Concatenazione delle feature e delle etichette
all_trainMFCC = [train_disco_features train_metal_features train_hiphop_features];
all_labelsMFCC = [labeldisco; labelmetal; labelhiphop];

% Normalizzazione delle feature MFCC
all_trainMFCC = all_trainMFCC';
mn = mean(all_trainMFCC);
st = std(all_trainMFCC);
all_trainMFCC =  (all_trainMFCC - repmat(mn,size(all_trainMFCC,1),1))./repmat(st,size(all_trainMFCC,1),1);

% Estrazione delle feature MFCC dai dati di test
disp('extract test features for MFCC...')
test_disco_features = extract_from_path(testdiscoPath,'wav',windowLength,stepLength);
test_metal_features = extract_from_path(testmetalPath,'wav',windowLength,stepLength);
test_hiphop_features = extract_from_path(testhiphopPath,'wav',windowLength,stepLength);

% Concatenazione delle feature e delle etichette di test
all_testMFCC = [test_disco_features test_metal_features test_hiphop_features];
testdiscoLabel = repmat(1,length(test_disco_features),1);
testmetalLabel = repmat(2,length(test_metal_features),1);
testhiphopLabel = repmat(3,length(test_metal_features),1);
ground_truthMFCC = [testdiscoLabel; testmetalLabel; testhiphopLabel];

% Normalizzazione dei dati di test MFCC
all_testMFCC = all_testMFCC';
all_testMFCC =  (all_testMFCC - repmat(mn,size(all_testMFCC,1),1))./repmat(st,size(all_testMFCC,1),1);

% Concatenazione delle feature CHROMA e MFCC
disp('link of MFCC + CHROMA')
all_testALL = [all_testCHROMA all_testMFCC];
all_trainALL =  [all_trainCHROMA all_trainMFCC];
all_labelsALL = all_labelsCHROMA;
all_ground_truthALL = ground_truthCHROMA;

% Addestramento del classificatore k-NN con diverse feature
k = [1 10 20];
all_kNN_rates=[];
[allRatesCHROMA, predicted_label_knn_chroma] = kNN_Training(all_trainCHROMA, all_labelsCHROMA, all_testCHROMA, ground_truthCHROMA, k, 1 , true);
all_kNN_rates = [all_kNN_rates; allRatesCHROMA];
[allRatesMFCC, predicted_label_knn_MFCC] = kNN_Training(all_trainMFCC, all_labelsMFCC, all_testMFCC, ground_truthMFCC, k, 2, true);
all_kNN_rates = [all_kNN_rates; allRatesMFCC];
[allRatesALL, predicted_label_knn_ALL] = kNN_Training(all_trainALL, all_labelsALL, all_testALL, all_ground_truthALL, k, 3, true);
all_kNN_rates = [all_kNN_rates; allRatesALL];

% Addestramento dell'albero decisionale per CHROMA
chroma_tree = fitctree(all_trainCHROMA, all_labelsCHROMA);
predicted_labelCHROMA = predict(chroma_tree, all_testCHROMA);

% Confusion matrix per CHROMA con k-NN e decision tree
figure(1)
subplot(2,1,1)
CKNN_chroma = confusionmat(ground_truthCHROMA, predicted_label_knn_chroma);
cmknn_chroma = confusionchart(CKNN_chroma, {'disco'  'metal' 'hiphop'}, 'Title', 'k-NN classification on CHROMA', 'RowSummary', 'row-normalized');
subplot(2,1,2)
CCHROMA = confusionmat(ground_truthCHROMA, predicted_labelCHROMA);
cmCHROMA = confusionchart(CCHROMA, {'disco'  'metal' 'hiphop'}, 'Title', 'DT+CHROMA classification', 'RowSummary', 'row-normalized');

% Stampa dei tassi di classificazione per CHROMA
disp('CHROMA rates')
rate_knnCHROMA = (CKNN_chroma(1,1)+CKNN_chroma(2,2)+CKNN_chroma(3,3))/sum(sum(CKNN_chroma)) * 100
rate_DTCHROMA = (CCHROMA(1,1)+CCHROMA(2,2)+CCHROMA(3,3))/sum(sum(CCHROMA)) * 100

% Addestramento dell'albero decisionale per MFCC
mfcc_tree = fitctree(all_trainMFCC, all_labelsMFCC);
predicted_labelMFCC = predict(mfcc_tree, all_testMFCC);

% Confusion matrix per MFCC con k-NN e decision tree
figure(2)
subplot(2,1,1)
CKNN = confusionmat(ground_truthMFCC, predicted_label_knn_MFCC);
cmknn = confusionchart(CKNN, {'disco'  'metal' 'hiphop'}, 'Title', 'k-NN classification on MFCC', 'RowSummary', 'row-normalized');
subplot(2,1,2)
CMFCC = confusionmat(ground_truthMFCC, predicted_labelMFCC);
cmMFCC = confusionchart(CMFCC, {'disco'  'metal' 'hiphop'}, 'Title', 'DT+MFCC classification', 'RowSummary', 'row-normalized');

% Stampa dei tassi di classificazione per MFCC
disp('MFCC rates')
rate_knnMFCC = (CKNN(1,1)+CKNN(2,2)+CKNN(3,3))/sum(sum(CKNN)) * 100
rate_DTMFCC = (CMFCC(1,1)+CMFCC(2,2)+CMFCC(3,3))/sum(sum(CMFCC)) * 100

% Addestramento dell'albero decisionale per MFCC+CHROMA
all_tree = fitctree(all_trainALL, all_labelsALL);
predicted_labelALL = predict(all_tree, all_testALL);

% Confusion matrix per CHROMA+MFCC con k-NN e decision tree
figure(3)
subplot(2,1,1)
CKNN_ALL = confusionmat(all_ground_truthALL, predicted_label_knn_ALL);
cmknn_ALL = confusionchart(CKNN_ALL, {'disco'  'metal' 'hiphop'}, 'Title', 'k-NN classification on MFCC + CHROMA', 'RowSummary', 'row-normalized');
subplot(2,1,2)
CALL = confusionmat(all_ground_truthALL, predicted_labelALL);
cmALL = confusionchart(CALL, {'disco'  'metal' 'hiphop'}, 'Title', 'DT+MFCC + CHROMA classification', 'RowSummary', 'row-normalized');

% Stampa dei tassi di classificazione per CHROMA+MFCC
disp('MFCC + CHROMA rates')
rate_knnALL = (CKNN_ALL(1,1)+CKNN_ALL(2,2)+CKNN_ALL(3,3))/sum(sum(CKNN_ALL)) * 100
rate_DTALL = (CALL(1,1)+CALL(2,2)+CALL(3,3))/sum(sum(CALL)) * 100

% Definizione delle matrici di tassi di classificazione per DT e k-NN
ALLrate_DTM = [rate_DTCHROMA 1; rate_DTMFCC 2; rate_DTALL 3];
ALLrate_KNN = [rate_knnCHROMA 1; rate_knnMFCC 2; rate_knnALL 3];

% Ricerca del miglior tasso di classificazione k-NN
disp("Searching for optimum k-NN rate")
% Cerca il massimo tasso di classificazione k-NN tra le caratteristiche
[~, knn_max_index] = max(all_kNN_rates(:,1));
disp(['k-NN Maximum Recognition Rate is: ' mat2str(all_kNN_rates(knn_max_index, 1))])

% Controlla da quale tipo di feature proviene il massimo tasso
if all_kNN_rates(knn_max_index, 2) == 1
    disp(['and it is achieved with CHROMA ' mat2str(all_kNN_rates(knn_max_index, 3)) ' Neighbors.']);
elseif all_kNN_rates(knn_max_index, 2) == 2
    disp(['and it is achieved with MFCC ' mat2str(all_kNN_rates(knn_max_index, 3)) ' Neighbors.']);
elseif all_kNN_rates(knn_max_index, 2) == 3
    disp(['and it is achieved with CHROMA + MFCC ' mat2str(all_kNN_rates(knn_max_index, 3)) ' Neighbors.']);
end

% Ricerca del miglior tasso di classificazione Decision Tree
disp("Searching for optimum DT rate")
[~, DT_max_index] = max(ALLrate_DTM(:,1));
disp(['DT Maximum Rate is: ' mat2str(ALLrate_DTM(DT_max_index, 1))])

% Controlla da quale tipo di feature proviene il massimo tasso
if ALLrate_DTM(DT_max_index, 2) == 1
    disp(['and it is achieved with CHROMA '])
elseif ALLrate_DTM(DT_max_index, 2) == 2
    disp(['and it is achieved with MFCC '])
elseif ALLrate_DTM(DT_max_index, 2) == 3
    disp(['and it is achieved with CHROMA + MFCC '])
end

% Aggiunta di rumore e denoising
add_noise_and_denoise(testdiscoPath, 'babble.wav', 5, 1);
add_noise_and_denoise(testmetalPath, 'babble.wav', 5, 2);
add_noise_and_denoise(testhiphopPath, 'babble.wav', 5, 3);

% Visualizzazione degli spettrogrammi per i file rumorosi
figure(4)
disp('plotting the spectrograms...')
noisyDiscofiles = dir('noisyfiles/disco/noisydisco*.wav');
for i = 1:length(noisyDiscofiles)
    [y, fs] = audioread(noisyDiscofiles(i).name);
    subplot(3,3,i)
    specgram(y, 1024, fs)
    title(noisyDiscofiles(i).name)
end

disp('plotting the spectrograms...')
noisyMetalfiles = dir('noisyfiles/metal/noisymetal*.wav');
for i = 1:length(noisyMetalfiles)
    [y, fs] = audioread(noisyMetalfiles(i).name);
    subplot(3,3,i+3)
    specgram(y, 1024, fs)
    title(noisyMetalfiles(i).name)
end

disp('plotting the spectrograms...')
noisyHiphopfiles = dir('noisyfiles/hiphop/noisyhiphop*.wav');
for i = 1:length(noisyHiphopfiles)
    [y, fs] = audioread(noisyHiphopfiles(i).name);
    subplot(3,3,i+6)
    specgram(y, 1024, fs)
    title(noisyHiphopfiles(i).name)
end

% Estrazione delle features dai file rumorosi
disp('extraction features of noisy train set')
test_disco_featuresNoisy = extract_from_path('C:\Users\furbe\OneDrive\Desktop\IAM\noisyfiles\disco\', 'wav', windowLength, stepLength);
test_metal_featuresNoisy = extract_from_path('C:\Users\furbe\OneDrive\Desktop\IAM\noisyfiles\metal\', 'wav', windowLength, stepLength);
test_hiphop_featuresNoisy = extract_from_path('C:\Users\furbe\OneDrive\Desktop\IAM\noisyfiles\hiphop\', 'wav', windowLength, stepLength);

% Etichette per le classi rumorose
testlabeldiscoNoisy = repmat(1, length(test_disco_featuresNoisy), 1);
testlabelmetalNoisy = repmat(2, length(test_metal_featuresNoisy), 1);
testlabelhiphopNoisy = repmat(3, length(test_hiphop_featuresNoisy), 1);

% Concatenazione delle features MFCC dei tre generi musicali
all_testMFCCNoisy = [test_disco_featuresNoisy test_metal_featuresNoisy test_hiphop_featuresNoisy];
ground_truthMFCCNoisy = [testlabeldiscoNoisy; testlabelmetalNoisy; testlabelhiphopNoisy];

% Normalizzazione delle features MFCC
all_testMFCCNoisy = all_testMFCCNoisy';
mn = mean(all_testMFCCNoisy);
st = std(all_testMFCCNoisy);
all_testMFCCNoisy = (all_testMFCCNoisy - repmat(mn, size(all_testMFCCNoisy, 1), 1)) ./ repmat(st, size(all_testMFCCNoisy, 1), 1);

% Estrazione delle features CHROMA dai file rumorosi
test_disco_featuresNoisy = extract_from_path_chroma('C:\Users\furbe\OneDrive\Desktop\IAM\noisyfiles\disco\', 'wav', windowLength, stepLength);
test_metal_featuresNoisy = extract_from_path_chroma('C:\Users\furbe\OneDrive\Desktop\IAM\noisyfiles\metal\', 'wav', windowLength, stepLength);
test_hiphop_featuresNoisy = extract_from_path_chroma('C:\Users\furbe\OneDrive\Desktop\IAM\noisyfiles\hiphop\', 'wav', windowLength, stepLength);

% Etichette per le classi rumorose (CHROMA)
testlabeldiscoNoisy = repmat(1, length(test_disco_featuresNoisy), 1);
testlabelmetalNoisy = repmat(2, length(test_metal_featuresNoisy), 1);
testlabelhiphopNoisy = repmat(3, length(test_hiphop_featuresNoisy), 1);

% Concatenazione delle features CHROMA dei tre generi musicali
all_testCHROMANoisy = [test_disco_featuresNoisy test_metal_featuresNoisy test_hiphop_featuresNoisy];
ground_truthCHROMANoisy = [testlabeldiscoNoisy; testlabelmetalNoisy; testlabelhiphopNoisy];

% Normalizzazione delle features CHROMA
all_testCHROMANoisy = all_testCHROMANoisy';
mn = mean(all_testCHROMANoisy);
st = std(all_testCHROMANoisy);
all_testCHROMANoisy = (all_testCHROMANoisy - repmat(mn, size(all_testCHROMANoisy, 1), 1)) ./ repmat(st, size(all_testCHROMANoisy, 1), 1);

% Unione delle features MFCC e CHROMA rumorose
disp('link of MFCC + CHROMA')
all_testALLNoisy = [all_testCHROMANoisy all_testMFCCNoisy];
all_labelsALL = all_labelsCHROMA;
all_ground_truthALLNoisy = ground_truthCHROMANoisy;

% Addestramento dell'albero decisionale per CHROMA (con dataset rumoroso)
chroma_tree = fitctree(all_trainCHROMA, all_labelsCHROMA);
predicted_labelCHROMANoisy = predict(chroma_tree, all_testCHROMANoisy);

% Creazione delle confusion matrix per k-NN e DT con dataset rumoroso (CHROMA)
figure(5)
subplot(2,1,1)
[allRatesCHROMANoisy, predicted_label_knn_chromaNoisy] = kNN_Training(all_trainCHROMA, all_labelsCHROMA, all_testCHROMANoisy, ground_truthCHROMANoisy, k, 1, true);
CKNN_chromaNoisy = confusionmat(ground_truthCHROMANoisy, predicted_label_knn_chromaNoisy);
cmknn_chromaNoisy = confusionchart(CKNN_chromaNoisy, {'disco', 'metal', 'hiphop'}, 'Title', 'k-NN based emotion classification on CHROMA NOISY SET', 'RowSummary', 'row-normalized');
subplot(2,1,2)
CCHROMANoisy = confusionmat(ground_truthCHROMANoisy, predicted_labelCHROMANoisy);
cmCHROMANoisy = confusionchart(CCHROMANoisy, {'disco', 'metal', 'hiphop'}, 'Title', 'DT+CHROMA music genre classification on CHROMA NOISY SET', 'RowSummary', 'row-normalized');

% Addestramento dell'albero decisionale per MFCC (con dataset rumoroso)
figure(6)
mfcc_tree = fitctree(all_trainMFCC, all_labelsMFCC);
predicted_labelMFCCNoisy = predict(mfcc_tree, all_testMFCCNoisy);
subplot(2,1,1)
[allRatesMFCCNoisy, predicted_label_knn_mfccNoisy] = kNN_Training(all_trainMFCC, all_labelsMFCC, all_testMFCCNoisy, ground_truthMFCCNoisy, k, 2, true);
MKNN_mfccNoisy = confusionmat(ground_truthMFCCNoisy, predicted_label_knn_mfccNoisy);
mmknn_mfccNoisy = confusionchart(MKNN_mfccNoisy, {'disco', 'metal', 'hiphop'}, 'Title', 'k-NN based emotion classification on MFCC NOISY SET', 'RowSummary', 'row-normalized');
subplot(2,1,2)
MMFCCNoisy = confusionmat(ground_truthMFCCNoisy, predicted_labelMFCCNoisy);
cmMFCCNoisy = confusionchart(MMFCCNoisy, {'disco', 'metal', 'hiphop'}, 'Title', 'DT+MFCC music genre classification on MFCC NOISY SET', 'RowSummary', 'row-normalized');

% Classificazione con features combinate (MFCC + CHROMA) su dataset rumoroso
figure(7)
[allRatesALLNoisy, predicted_label_knn_ALLNoisy] = kNN_Training(all_trainALL, all_labelsALL, all_testALLNoisy, all_ground_truthALLNoisy, k, 3, true);
mfcc_treeALLNoisy = fitctree(all_trainALL, all_labelsALL);
predicted_labelALLNoisy = predict(mfcc_treeALLNoisy, all_testALLNoisy);
subplot(2,1,1)
CKNNALLNoisy = confusionmat(all_ground_truthALLNoisy, predicted_label_knn_ALLNoisy);
cmknnALLNoisy = confusionchart(CKNNALLNoisy, {'disco', 'metal', 'hiphop'}, 'Title', 'k-NN based emotion classification on MFCC + CHROMA NOISY SET', 'RowSummary', 'row-normalized');
subplot(2,1,2)
CALLNoisy = confusionmat(all_ground_truthALLNoisy, predicted_labelALLNoisy);
cmALLNoisy = confusionchart(CALLNoisy, {'disco', 'metal', 'hiphop'}, 'Title', 'DT+MFCC+CHROMA music genre classification NOISY SET', 'RowSummary', 'row-normalized');





