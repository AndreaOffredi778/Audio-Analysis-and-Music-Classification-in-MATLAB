function [allRates, predicted_label_knn] = kNN_Training(all_train, all_labels, all_test, ground_truth, k, type, flag_disp)
    % Initialize the array for recognition rates
    allRates = [];
    
    % Loop through each k to train and test kNN model
    for kk = 1:length(k)
        disp(['Set-up the kNN... number of neighbors: ', mat2str(k(kk))]);
        
        % Train the kNN model
        Mdl_chroma = fitcknn(all_train, all_labels', 'NumNeighbors', k(kk));
        
        % Test the kNN model
        predicted_label = predict(Mdl_chroma, all_test);
        
        % Measure the performance
        correct = 0;
        for i = 1:length(predicted_label)
            if predicted_label(i) == ground_truth(i)
                correct = correct + 1;
            end
        end

         rate(kk) = (correct/length(predicted_label))*100;
    if flag_disp == true 
        disp(['Recognition Rate with ' mat2str(k(kk)) ' Neighbors: ' mat2str(rate(kk))])
    end
    array_rate = [rate(kk) type k(kk)]; 
    allRates = [allRates; array_rate];
    end
    
    % Find the maximum rate and corresponding k
    [max_rate, b] = max(rate);
    best_k = k(b);
    
    if type == 1
        s = sprintf('CHROMA:');
    elseif type == 2
        s = sprintf('MFCC:');
    elseif type == 3
        s = sprintf('CHROMA + MFCC:');
    end

    disp([newline 'Recap ' s]) %stampa dei rate calcolati dei diversi valori k
    for i = 1:length(k)
        disp([mat2str(k(i)) ' -> ' mat2str(rate(i))])
    end
    
    % Train using the best-performing k for kNN
    knn_Mdl_chroma = fitcknn(all_train, all_labels', 'NumNeighbors', best_k);
    predicted_label_knn = predict(knn_Mdl_chroma, all_test);
    
    % Display the best-performing k and max recognition rate
    disp(['Best-performing k: ', num2str(best_k)]);
    disp(['Maximum recognition rate: ', num2str(max_rate), '%']);
end
