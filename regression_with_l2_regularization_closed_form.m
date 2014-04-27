% Linear Regression with L2 Regularization
% Closed-form solution
% Author: Amir Azarbakht <azarbaam@eecs.oregonstate.edu>
% Date: 2014-04-15


% function [lambdaSet, SSE_train, SSE_test] = regression_with_L2_regularization(fileName)

clear all;
close all;
clc;

% if nargin == 1
%    data = csvread('[fileName]');
% elseif nargin == 0
%     data = csvread('train.csv');
% else
%     disp('No input data file specified');
% end

% import the data
data = csvread('train.csv');

% Normalize the data into a [0,1] range
for j = 2:(size(data,2)),
    minFeatureVec = min(data(:,j));
    maxFeatureVec = max(data(:,j));
    for i = 1:size(data,1),
        if maxFeatureVec ~= minFeatureVec
            data(i,j) = (data(i,j) - minFeatureVec)/abs(maxFeatureVec - minFeatureVec);
        else
            disp('Normalization Error! Min and Max of a feature vector are equal. ');
        end
    end
end

Y = data(:,46);
X = data(:,1:45);



data_test = csvread('test.csv');
clear minFeatureVec maxFeatureVec;

% Normalize the data into a [0,1] range
for j = 2:(size(data_test,2)),
    minFeatureVec = min(data_test(:,j));
    maxFeatureVec = max(data_test(:,j));
    for i = 1:size(data_test,1),
        if maxFeatureVec ~= minFeatureVec
            data_test(i,j) = (data_test(i,j) - minFeatureVec)/abs(maxFeatureVec - minFeatureVec);
        else
            disp('Normalization Error! Min and Max of a feature vector are equal. ')
        end
    end
end

Y_test = data_test(:,46);
X_test = data_test(:,1:45);

% Lambda: Regularization term. Trying varying lambda values to choose the best lambda value based on its corresponding SSE classification error. 
lambdaSet = [0, 0.001, 0.01, 0.1, 1, 2, 4, 5, 10, 15, 25, 40, 60, 85, 100];

[dimX1, dimX2] = size(X);
[dimL1, dimL2] = size(lambdaSet);

SSE_train = Inf(size(lambdaSet));
SSE_test = Inf(size(lambdaSet));
w = zeros(dimX2,dimL2);

for j = 1: dimL2,
    lambda = lambdaSet(1,j);
    % calculate w
    w(:,j) = (inv((lambda * eye(size(X' * X))) + X' * X)) * X' * Y;
    
    % calculate Sum of Squared Error for the training set
    SUMSSE_train = 0;
    SUMSSE_test = 0;
    
    for  i = 1:dimX1,
        SUMSSE_train = SUMSSE_train + ((( w(:,j)' * X(i,:)') - Y(i,1))^2);
        % calculate Sum of Squared Error for the test set
        SUMSSE_test = SUMSSE_test + (((w(:,j)' * X_test(i,:)') - Y_test(i,1))^2);
    end
    SSE_train(1,j) = 0.5 * (SUMSSE_train);
    SSE_test(1,j) = 0.5 * (SUMSSE_test);    
end

% SSE_train
% SSE_test

% save the SSE results as a file

% csvwrite('SSE_train.csv', SSE_train);
% csvwrite('SSE_test.csv', SSE_test);
fig1_lasso = [lambdaSet; SSE_train; SSE_test;];
csvwrite('fig1_lasso.csv', fig1_lasso);


% Plot the SSE vs. lambda
figure1 = figure('Color',[1 1 1]);
axes1 = axes('Parent',figure1,...
    'XTickLabel',{'0.001','0.01','0.1','1','2','4','5','10','15','25','40','60','85','100'},...
    'XTick',[0 1 2 3 4 5 6 7 8 9 10 11 12 13]);
box(axes1,'on');
hold(axes1,'all');
plot((0:1:(size(SSE_train,2)-2)), SSE_train(2:end), 'b');
hold on;
plot((0:1:(size(SSE_test,2)-2)), SSE_test(2:end), 'r');
title({'Classification Error (SSE) vs. Regularization Term (Lambda)'});
legend('SSE Train', 'SSE Test');
xlabel('Lambda Values');
ylabel('Sum of Squared Error');

% save as PNG and EPS
saveas(1, 'Regression_Classification_Error_SSE_vs_Regularization_Term_Lambda', 'epsc2');
saveas(1, 'Regression_Classification_Error_SSE_vs_Regularization_Term_Lambda', 'png');

% saveas(1, 'Regression_with_lambda-0_Classification_Error_SSE_vs_Regularization_Term_Lambda', 'eps');
% saveas(1, 'Regression_with_lambda-0_Classification_Error_SSE_vs_Regularization_Term_Lambda', 'png');


% [a,b] = min(SSE_test)
% [a,b] = max(SSE_test)
% [a,b] = min(SSE_train)
