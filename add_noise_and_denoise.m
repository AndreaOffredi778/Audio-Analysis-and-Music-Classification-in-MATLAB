function add_noise_and_denoise(cleanDir, noiseFileName, SNR, type)  
    % Ottieni i file nella directory specificata
    cleanFiles = dir(fullfile(cleanDir, '*.wav')); % Filtra solo i file .wav
    cleanFileNames = {cleanFiles.name}; % Estrai i nomi dei file

    % Carica il file del rumore
    [noiseFile, fs2] = audioread(noiseFileName);
    
    % Cicla su tutti i file clean
    for i = 1:length(cleanFileNames)
        % Visualizza il file in elaborazione
        disp(['Adding noise ', noiseFileName, ' to file ', cleanFileNames{i}, ' at SNR ', mat2str(SNR), '...']);
        
        % Leggi il file audio clean
        [x1, fs] = audioread(fullfile(cleanDir, cleanFileNames{i}));
        
        % Allinea la lunghezza del file del rumore alla lunghezza del file clean
        noiseFileTrimmed = noiseFile(1:length(x1));
        
        % Crea il segnale con rumore
        noisy_sig = sigmerge(x1, noiseFileTrimmed, SNR);

         if type == 1
        s = sprintf('disco');
        elseif type == 2
        s = sprintf('metal');
        elseif type == 3
        s = sprintf('hiphop');
        end
        
        % Crea il nome del file con rumore
        noisyFile = ['noisy', cleanFileNames{i}(1:end-4), '_SNR', mat2str(SNR), '_noise_', noiseFileName(1:end-4), '.wav'];
        noisyfilesDir = 'C:\Users\furbe\OneDrive\Desktop\IAM\noisyfiles\';
        noisyfilesDir = [noisyfilesDir,s];
        noisyFilePath = fullfile(noisyfilesDir, noisyFile);
        % Salva il file con rumore
        disp('Saving the noisy file...');
        audiowrite(noisyFilePath, noisy_sig, fs);
        
        % Crea il nome del file denoised (enhanced)
        enhFile = ['enhanced_', cleanFileNames{i}(1:end-4), '_SNR', mat2str(SNR), '_noise_', noiseFileName(1:end-4), '.wav'];
        
        % Esegui la denoising
        disp('Denoising...');
        specsub(noisyFile, enhFile);
    end
    
    disp('Processing complete!');
end