facial_attributes <- read.csv("C:/Users/Quek Wei Liang/OneDrive/Documents/National University of Singapore/AY2018-19 Year 3 Semester 2/DSC3216 Predictive Analytics in Business/Projects/Datasets/list_attr_celeba.csv")
#Make a column
x <- cbind(Feature_reduction_normalised$gender, Feature_reduction_normalised$label)
View(x)
***********************************************************************************************
#Check proportion of class variables
prop.table(table(normalised_data_with_labels$label))
**********************************************************************************************
#Create random sample
set.seed(123)
train_sample <- sample(nrow(Feature_reduction_normalised),nrow(Feature_reduction_normalised)*0.7)
str(train_sample)
print(train_sample)
trainData <- Feature_reduction_normalised [train_sample,]
testData <- Feature_reduction_normalised [-train_sample,]

set.seed(123)
train_sample <- sample(nrow(Feature_reduction_not_normalised),nrow(Feature_reduction_not_normalised)*0.7)
str(train_sample)
print(train_sample)
trainData <- Feature_reduction_not_normalised [train_sample,]
testData <- Feature_reduction_not_normalised [-train_sample,]

train_sample$label <- as.factor(train_sample$label)
test_sample$label <- as.factor(test_sample$label)
**********************************************************************************************
#Logistic Regression
https://stats.idre.ucla.edu/r/dae/multinomial-logistic-regression/
require(foreign)
require(nnet)
require(ggplot2)
require(reshape2)
require(ISLR)
names(Feature_reduction_normalised)
head(Feature_reduction_normalised)
summary(Feature_reduction_normalised)
facial_logreg <- glm(label ~., data = Feature_reduction_normalised, family = binomial, control = list(maxit = 50))
summary(facial_logreg)
trainPred_logreg <- predict(facial_logreg, newdata = trainData, type = 'response')
logsregPred_train <- ifelse(trainPred_logreg > 0.5,1,0)
confusionMatrix(table(logsregPred_train, trainData$label))
testPred_logreg <- predict(facial_logreg, newdata = testData, type = 'response')
logsregPred_test <- ifelse(testPred_logreg > 0.5,1,0)
confusionMatrix(table(logsregPred_test, testData$label))

library(rms)
null_facial_logreg <- glm(label~1, family = binomial, data = Feature_reduction_normalised )
1-logLik(facial_logreg)/logLik(null_facial_logreg)

summary(facial_logreg)$r.squared

require(foreign)
require(nnet)
require(ggplot2)
require(reshape2)
require(ISLR)
library(caret)
library(e1071)
facial_logreg <- glm(label ~., data = trainData, family = binomial, control = list(maxit = 50))
summary(facial_logreg)
trainPred_logreg <- predict(facial_logreg, newdata = trainData, type = 'response')
logsregPred_train <- ifelse(trainPred_logreg > 0.5,1,0)
confusionMatrix(table(logsregPred_train, trainData$label))
testPred_logreg <- predict(facial_logreg, newdata = testData, type = 'response')
logsregPred_test <- ifelse(testPred_logreg > 0.5,1,0)
confusionMatrix(table(logsregPred_test, testData$label))
**********************************************************************************************
#Classificiation Tree
library(party)
trainData$label <- as.factor(trainData$label)
facial_ctree <- ctree(label ~ ., data = trainData)
print(facial_ctree)
plot(facial_ctree, type = 'simple')
plot(facial_ctree)
table(predict(facial_ctree), trainData$label)
testData$label <- as.factor(testData$label)
facial_ctree_test <- ctree(label ~ ., data = testData)
print(facial_ctree_test)
table(predict(facial_ctree_test), testData$label)

library(party)
trainData$label <- as.factor(trainData$label)
facial_ctree <- ctree(label ~ ., data = trainData)
print(facial_ctree)
plot(facial_ctree, type = 'simple')
plot(facial_ctree)
table(predict(facial_ctree), trainData$label)
testData$label <- as.factor(testData$label)
facial_ctree_test <- ctree(label ~ ., data = testData)
print(facial_ctree_test)
table(predict(facial_ctree_test), testData$label)

normalised_data_with_labels$before_gender <- as.factor(normalised_data_with_labels$before_gender) #Convert your character values to factors. 
normalised_data_with_labels$after_gender <- as.factor(normalised_data_with_labels$after_gender)
same_ctreePred_train <- trainPred_ctree == trainData$label
table(same_ctreePred_train)
trainPred_ctree <- predict(facial_ctree, newdata = trainData, type = 'response')
ctreePred_train <- ifelse(trainPred_ctree > 0.5,1,0)
table(predict(ctreePred_train), trainData$label)
testData$label <- as.factor(testData$label)
testPred_ctree <- predict(facial_ctree, newdata = testData, type = 'response')
ctreePred_test <- ifelse(testPred_ctree > 0.5,1,0)
table(testPred_ctree, testData$label)
**********************************************************************************************
#SVM
library(e1071)
library(gmodels)
svm_classifier_linear = svm(formula = label~., data = trainData, type= "C-classification", kernel = "linear")
svm_pred_linear = predict(svm_classifier_linear, newdata= testData)
table(svm_pred_linear, testData$label)
CrossTable(testData$label,svm_pred_linear,prop.chisq= FALSE, prop.c= FALSE, prop.r = FALSE)
confusionMatrix(table(svm_pred_linear, testData$label))

svm_classifier_polynomial = svm(formula = label~., data = trainData, type= "C-classification", kernel = "polynomial")
svm_pred_polynomial = predict(ssvm_classifier_polynomial, newdata= testData)
table(svm_pred_polynomial, testData$label)
CrossTable(testData$label,svm_pred_polynomial,prop.chisq= FALSE, prop.c= FALSE, prop.r = FALSE)
confusionMatrix(table(svm_pred_polynomial, testData$label))

svm_classifier_radial = svm(formula = label~., data = trainData, type= "C-classification", kernel = "radial")
svm_pred_radial = predict(svm_classifier_radial, newdata= testData)
table(vm_pred_radial, testData$label)
CrossTable(testData$label,svm_pred_radial,prop.chisq= FALSE, prop.c= FALSE, prop.r = FALSE)
confusionMatrix(table(svm_pred_radial, testData$label), mode = "prec_recall", positive = "1")

svm_classifier_sigmoid = svm(formula = label~., data = trainData, type= "C-classification", kernel = "sigmoid")
svm_pred_sigmoid = predict(svm_classifier_sigmoid, newdata= testData)
table(svm_pred_sigmoid , testData$label)
CrossTable(testData$label,svm_pred_sigmoid ,prop.chisq= FALSE, prop.c= FALSE, prop.r = FALSE)
confusionMatrix(table(svm_pred_sigmoid, testData$label))

library(kernlab)
same_svmPred_train <- trainPred_svm == trainData$label
table(same_svmPred_train)
facial_svm <- ksvm(label ~., data = trainData, kernel ="vanilladot")
summary(facial_svm)
trainData$label <- as.factor(trainData$label)
trainPred_svm <- predict(facial_svm, newdata = trainData, type = 'response')
confusionMatrix(table(predict(trainPred_svm), trainData$label))
testData$label <- as.factor(testData$label)
testPred_svm <- predict(facial_svm, newdata = testData, type = 'response')
confusionMatrix(table(testPred_svm, testData$label))

#besseldot kernel#
facial_svm <- ksvm(label ~ before_age + before_gender	+ before_baldness	+ before_beard	+ before_moustache	+ before_sideburns	+ before_pupilLeft_x	+ before_pupilLeft_y	+ before_pupilRight_x	+ before_pupilRight_y	+ before_NoseTip_x	+ before_NoseTip_y	+ before_noseRootLeft_x	+ before_noseRootLeft_y	+ before_noseRootRight_x	+ before_noseRootRight_y	+ before_underLipBottom_x	+ before_underLipBottom_y	+ before_underLipTop_x	+ before_underLipTop_y +	before_upperLipBottom_x	+ before_upperLipBottom_y	+ before_upperLipTop_x	+ before_upperLipTop_y	+ after_age	+ after_gender + after_baldness	+ after_beard	+ after_moustache	+ after_sideburns	+ after_pupilLeft_x	+ after_pupilLeft_y	+ after_pupilRight_x	+ after_pupilRight_y	+ after_NoseTip_x	+ after_NoseTip_y	+ after_noseRootLeft_x	+ after_noseRootLeft_y	+ after_noseRootRight_x	+ after_noseRootRight_y	+ after_underLipBottom_x	+ after_underLipBottom_y	+ after_underLipTop_x	+ after_underLipTop_y	+ after_upperLipBottom_x	+ after_upperLipBottom_y	+ after_upperLipTop_x	+ after_upperLipTop_y, data = trainData, kernel ="besseldot")

summary(svm_classifier)
r.squared(svm_classifier)
library(plm)
***********************************************************************************************  
# Cross Validation (Cross-validation is also known as a resampling method because it involves fitting the same statistical method multiple times using different subsets of the data.)
#Manual k-fold cross validation - lda represent Linear Discriminant Function with Jacknifed Prediction (i.e., leave one out)
https://scikit-learn.org/stable/modules/cross_validation.html
library(MASS)
set.seed(123)
folds <- sample(rep(1:10, length.out = nrow(Feature_reduction_normalised)), size = nrow(Feature_reduction_normalised), replace = F)
table(folds)
folds
CV_lda <- lapply(1:10, function(x){ 
  model <- lda(label ~ ., Feature_reduction_normalised[folds != x, ])
  preds <- predict(model,  Feature_reduction_normalised[folds == x,], type="response")$class
  return(data.frame(preds, real = Feature_reduction_normalised$label[folds == x]))
})
CV_lda <- do.call(rbind, CV_lda)
confusionMatrix(table(CV_lda$preds, CV_lda$real))

###Jie Da's Cross Validation###
library(plyr)
# CV using boot
library(boot)
# Cost function for a binary classifier suggested by boot package
facial_logreg <- glm(label ~., data = Feature_reduction_normalised, family = binomial, control = list(maxit = 50))
facial_ctree_test <- ctree(label ~ ., data = testData)
svm_classifier_radial = svm(formula = label~., data = trainData, type= "C-classification", kernel = "radial")

# (otherwise MSE is the default)
cost <- function(r, pi = 0) mean(abs(r-pi) > 0.5)

# K-fold CV K=10 (accuracy)

library(ROCR)
cross_validation = function(nbfolds){
  perf = data.frame()
  #create folds
  folds = createFolds(trainData$label, nbfolds, list = TRUE, returnTrain = TRUE)
  
  #loop nbfolds times to find optimal threshold
  for(i in 1:nbfolds)
  {
    #train the model on part of the data
    model = glm(label~., data=trainData[folds[[i]],], family = "binomial", control = list(maxit = 50))
    
    #validate on the remaining part of the data
    probs = predict(model, type="response", newdata = trainData[-folds[[i]],])
    
    #Threshold selection based on Accuracy
    #create a prediction object based on the predicted values
    pred = prediction(probs,trainData[-folds[[i]],]$label)
    
    #measure performance of the prediction
    acc.perf = performance(pred, measure = "acc")
    
    #Find index of most accurate threshold and add threshold in data frame
    ind = which.max( slot(acc.perf, "y.values")[[1]] )
    acc = slot(acc.perf, "y.values")[[1]][ind]
    optimalThreshold = slot(acc.perf, "x.values")[[1]][ind]
    row = data.frame(threshold = optimalThreshold, accuracy = acc)
    
    #Store the best thresholds with their performance in the perf dataframe
    perf = rbind(perf, row)
  }
  #Get the threshold with the max accuracy among the nbfolds and predict based on it on the unseen test set
  indexOfMaxPerformance = which.max(perf$accuracy)
  optThresh = perf$threshold[indexOfMaxPerformance]
  probs = predict(model, type="response", newdata = testData)
  predictions = data.frame(label=testData$label, pred=probs)
  T = table(predictions$label, predictions$pred)
  F1 = (2*(T[1,1]))/((2*(T[1,1]))+T[2,1]+T[1,2])
  F1
}
cross_validation(5)
  

1-cv.glm(Feature_reduction_normalised,facial_logreg,K=10,cost=cost)$delta[1]

cross_validation_tree = function(nbfolds){
  perf = data.frame()
  #create folds
  folds = createFolds(trainData$label, 5, list = TRUE, returnTrain = TRUE)
  
  #loop nbfolds times to find optimal threshold
for(i in 1:5){
    #train the model on part of the data
    model = ctree(label ~ ., data = trainData)
    
    #validate on the remaining part of the data
    probs = predict(model, type="response", newdata = trainData[-folds[[i]],])
    
    #Threshold selection based on Accuracy
    #create a prediction object based on the predicted values
    pred = table(probs,trainData[-folds[[i]],]$label)
    
    #measure performance of the prediction
    acc.perf = performance(pred, measure = "acc")
    
    #Find index of most accurate threshold and add threshold in data frame
    ind = which.max( slot(acc.perf, "y.values")[[1]] )
    acc = slot(acc.perf, "y.values")[[1]][ind]
    optimalThreshold = slot(acc.perf, "x.values")[[1]][ind]
    row = data.frame(threshold = optimalThreshold, accuracy = acc)
    
    #Store the best thresholds with their performance in the perf dataframe
    perf = rbind(perf, row)
  }
  #Get the threshold with the max accuracy among the nbfolds and predict based on it on the unseen test set
  indexOfMaxPerformance = which.max(perf$accuracy)
  probs = predict(model, type="response", newdata = testData)
  predictions = data.frame(label=testData$label, pred=probs)
  T = table(predictions$label, predictions$pred)
  F1 = (2*(T[1,1]))/((2*(T[1,1]))+T[2,1]+T[1,2])
  F1
}
cross_validation_tree(5)

1-cv.glm(Feature_reduction_normalised,facial_ctree_test,K=10,cost=cost)$delta[1]
1-cv.glm(Feature_reduction_normalised,svm_classifier_radial,K=10,cost=cost)$delta[1]



##SVM##
https://stackoverflow.com/questions/20461476/svm-with-cross-validation-in-r-using-caret
library(caret)
ctrl <- trainControl(method = "cv", savePred=T, classProb=T)
set.seed(123)
confusionMatrix(table(CV_svm$preds_svm, CV_svm$real))

#Other cross validation methods
http://www.sthda.com/english/articles/38-regression-model-validation/157-cross-validation-essentials-in-r/
https://stackoverflow.com/questions/49051314/how-to-produce-a-confusion-matrix-from-cross-validation
library(tidyverse)
library(caret)

###The Validation Set Approach###
# Split the data into training and test set
set.seed(123)
training.samples <- Feature_reduction_normalised$label %>%
  createDataPartition(p = 0.7, list = FALSE)
train.data  <- Feature_reduction_normalised [training.samples, ]
test.data <- Feature_reduction_normalised [-training.samples, ]
# Build the model
model <- lm(label ~., data = train.data)
# Make predictions and compute the R2, RMSE and MAE
predictions <- model %>% predict(test.data)
data.frame( R2 = R2(predictions, test.data$label),
            RMSE = RMSE(predictions, test.data$label),
            MAE = MAE(predictions, test.data$label))
#When comparing two models, the one that produces the lowest test sample RMSE is the preferred model.
#the RMSE and the MAE are measured in the same scale as the outcome variable. Dividing the RMSE by the average value of the outcome variable will give you the prediction error rate, which should be as small as possible:
RMSE(predictions, test.data$label)/mean(test.data$label)

###Leave one out cross validation - LOOCV###
# Define training control
train.control <- trainControl(method = "LOOCV")
# Train the model
model <- train(label ~., data = Feature_reduction_normalised, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

###K-fold cross-validation###
# Define training control
set.seed(123) 
train.control <- trainControl(method = "cv", number = 5)
# Train the model
model <- train(label ~., data = Feature_reduction_normalised, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

###K-fold cross-validation### with NB
# Define training control
set.seed(123) 
train.control <- trainControl(method = "cv", number = 5)
# Train the model
model <- train(label ~., data = Feature_reduction_normalised, method = "nb",
               trControl = train.control)
# Summarize the results
print(model)

###Repeated K-fold cross-validation###
# Define training control
set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 5, repeats = 20)
# Train the model
model <- train(label ~., data = Feature_reduction_normalised, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

#k-fold Cross Validation (Linear Discriminant Function with Jacknifed Prediction (i.e., leave one out))
```{r}
library(boot)
# Cost function for a binary classifier suggested by boot package
facial_logreg <- glm(label ~., data = Feature_reduction_normalised, family = binomial, control = list(maxit = 50))
facial_ctree_test <- ctree(label ~ ., data = testData)
svm_classifier_radial = svm(formula = label~., data = trainData, type= "C-classification", kernel = "radial")

# (otherwise MSE is the default)
cost <- function(r, pi = 0) mean(abs(r-pi) > 0.5)

# K-fold CV K=10 (accuracy)

1-cv.glm(Feature_reduction_normalised,facial_logreg,K=10,cost=cost)$delta[1]
1-cv.glm(Feature_reduction_normalised,facial_ctree_test,K=10,cost=cost)$delta[1]
1-cv.glm(Feature_reduction_normalised,svm_classifier_radial,K=10,cost=cost)$delta[1]
```
**********************************************************************************************  
# F1-score
https://stackoverflow.com/questions/8499361/easy-way-of-counting-precision-recall-and-f1-score-in-r
https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/
https://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html#perclass
###Log Regression###
library(MLmetrics)
logreg <- glm(formula = label ~ .,
              family = binomial(link = "logit"), data = Feature_reduction_normalised)
pred_log <- ifelse(logreg$fitted.values < 0.5, 0, 1)
F1_Score(y_pred = pred_log, y_true = Feature_reduction_normalised$label, positive = "0")
F1_Score(y_pred = pred_log, y_true = Feature_reduction_normalised$label, positive = "1")

##Classification Tree##
library(party)
y <- testData$label
y.factor_ctree <- factor(y)
predictions_ctree <- facial_ctree_test
prediction.factor_ctree <- factor(predictions_ctree)
precision_ctree <- posPredValue(predictions.factor_ctree, y.factor_ctree, positive="1")
recall_ctree <- sensitivity(predictions.factor_ctree,y.factor_ctree, positive="1")

F1_ctree <- ( 2* precision_ctree * recall_ctree) / (precision_ctree + recall_ctree)
F1_ctree

##SVM##
library(caret)
y <- testData$label
y.factor_svm <- factor(y)
predictions_svm <- svm_pred_radial 
predictions.factor_svm <- factor(predictions_svm)
precision_svm <- posPredValue(predictions.factor_svm, y.factor_svm, positive="1")
recall_svm <- sensitivity(predictions.factor_svm,y.factor_svm, positive="1")

F1_svm <- ( 2* precision_svm * recall_svm) / (precision_svm + recall_svm)
F1_svm

###TEMPLATE###
# From confusion matrix, Sensitivity (also known as recall) and Pos Pred Value(also known as precision)
y <- ... # factor of positive / negative cases
predictions <- ... # factor of predictions

precision <- posPredValue(predictions, y, positive="1")
recall <- sensitivity(predictions, y, positive="1")

F1 <- (2 * precision * recall) / (precision + recall) 
**********************************************************************************************
#Multiple ROC curve
https://stackoverflow.com/questions/50834314/format-of-prediction-for-rocr-curves-in-r
library(ggplot2)
library(ROCR)

logreg_ROC <- glm(label~., data=Feature_reduction_normalised, family="binomial")
pred_logreg_1 <- prediction(predict(logreg_ROC), Feature_reduction_normalised$label)
perf_logreg_ROC <- performance(pred_logreg_1,"tpr","fpr")
ctree_ROC <- ctree(label ~ ., data = Feature_reduction_normalised)
pred_ctree_2 <- prediction(predict(ctree_ROC), Feature_reduction_normalised$label)
perf_ctree_ROC <- performance(pred_ctree_2,"tpr","fpr")
svm_ROC <- svm(formula = label~., data = Feature_reduction_normalised, type= "C-classification", kernel = "radial")
predict_svm <- predict(svm_ROC, type = 'response')
predict_svm
pred_svm_3 <- prediction(as.numeric(predict_svm), as.numeric(Feature_reduction_normalised$label))
perf_svm_ROC <- performance(pred_svm_3, 'tpr','fpr')

plot(perf_logreg_ROC, col=4)
plot(perf_ctree_ROC, add = TRUE, col=7)
plot(perf_svm_ROC, add = TRUE, col = 3)
  legend('topright',
       c('logreg','svm','ctree'),
       fill=topo.colors(3))
  
#Single ROC curve
https://cran.r-project.org/web/packages/plotROC/vignettes/examples.html
https://stackoverflow.com/questions/28443834/how-to-plot-a-roc-curve-using-rocr-package-in-r-with-only-a-classification-con
library(ggplot2)
library(pROC)
library(ROCR)
mod <- glm(label~., data=Feature_reduction_normalised, family="binomial")
pred1 <- prediction(predict(mod), Feature_reduction_normalised$label)
perf1 <- performance(pred1,"tpr","fpr")
plot(perf1)
************************************************************************************************
#resize the images
ggsave("overall.png", plot = last_plot(), path = "~/Desktop/Project", width = 30, height = 25, units = "cm", dpi = 300)

ggsave("DSC3216_Project_ctree.png", plot = last_plot(), path = "~/Desktop", width = 100, height = 50, units = "cm", dpi = 500)
*************************************************************************************************
###Speed test###
facial_logreg_speed <- glm(label ~., data = Feature_reduction_normalised, family = binomial, control = list(maxit = 50))
cat('Logreg:', facial_logreg_speed)
facial_ctree_speed <- ctree(label ~ ., data = Feature_reduction_normalised)
cat('ctree:', facial_ctree_speed)
facial_svm_speed <- svm(formula = label~., data = Feature_reduction_normalised, type= "C-classification", kernel = "radial")
cat('SVM:', facial_svm_speed)
speed_test <- microbenchmark('Logreg' = {glm(label ~., data = Feature_reduction_normalised, family = binomial, control = list(maxit = 50))},
                     'ctree' = {ctree(label ~ ., data = Feature_reduction_normalised)},
                     'SVM' = {svm(formula = label~., data = Feature_reduction_normalised, type= "C-classification", kernel = "radial")}
)

boxplot(speed_test, unit='ns',log=F,horizontal=T,col=topo.colors(5))
legend('topright',
       title='Methods',
       c('seq(by)','seq(length.out)','seq.int(by)','seq.int(length.out)','c()'), 
       fill=topo.colors(5)
)
***************************************************************************************************
#Learning Curve
https://www.r-bloggers.com/learning-from-learning-curves/
https://stackoverflow.com/questions/38582076/how-to-plot-a-learning-curve-in-r

library(caret)
# set seed for reproducibility
set.seed(123)

# randomize dataset and split dataset into training and test sets
train_sample <- sample(nrow(Feature_reduction_normalised),nrow(Feature_reduction_normalised)*0.7)
str(train_sample)
trainData <- Feature_reduction_normalised [train_sample,]
testData <- Feature_reduction_normalised [-train_sample,]

# create empty data frame 
learnCurve <- data.frame(m = integer(21),
                         trainRMSE = integer(21),
                         cvRMSE = integer(21))

# test data response feature
testY <- Feature_reduction_normalised$label

# Run algorithms using 5-fold cross validation and 3 repeats
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# loop over training examples
for (i in 3:21) {
  learnCurve$m[i] <- i
  
  # train learning algorithm with size i
  fit.lm <- train(label~., data= Feature_reduction_normalised, method="lm", metric=metric,
                  preProc=c("center", "scale"), trControl=trainControl)        
  learnCurve$trainRMSE[i] <- fit.lm$results$RMSE
  
  # use trained parameters to predict on test data
  prediction <- predict(fit.lm, newdata = testData)
  rmse <- postResample(prediction, testY)
  learnCurve$cvRMSE[i] <- rmse[1]
}


# plot learning curves of training set size vs. error measure
# for training set and test set
plot(log(learnCurve$trainRMSE),type = "o",col = "red", xlab = "Training set size",
     ylab = "Error (RMSE)", main = "Linear Model Learning Curve")
lines(log(learnCurve$cvRMSE), type = "o", col = "blue")
legend('topright', c("Train error", "Test error"), lty = c(1,1), lwd = c(2.5, 2.5),
       col = c("red", "blue"))

dev.off()

#Learning Curve (Still working on it)
```{r}
library(caret)
set.seed(123)
train_sample <- sample(nrow(Feature_reduction_normalised),nrow(Feature_reduction_normalised)*0.7)
str(train_sample)
trainData <- Feature_reduction_normalised [train_sample,]
testData <- Feature_reduction_normalised [-train_sample,]

learnCurve <- data.frame(m = integer(21),
                         trainRMSE = integer(21),
                         testRMSE = integer(21),
                         cvRMSE = integer(21))

testY <- Feature_reduction_normalised$label

trainControl <- trainControl(method="repeatedcv", number=5, repeats=3)
metric <- "RMSE"

for (i in 3:21) {
  learnCurve$m[i] <- i
  
  fit.lm <- train(label~., data= Feature_reduction_normalised, method="lm", metric=metric,
                  preProc=c("center", "scale"), trControl=trainControl)        
  learnCurve$trainRMSE[i] <- fit.lm$results$RMSE
  
  prediction <- predict(fit.lm, newdata = testData)
  rmse <- postResample(prediction, testY)
  learnCurve$cvRMSE[i] <- rmse[1]
}

plot(log(learnCurve$trainRMSE),type = "o",col = "red", xlab = "Training set size",
     ylab = "Error (RMSE)", main = "Linear Model Learning Curve")
lines(log(learnCurve$cvRMSE), type = "o", col = "blue")
legend('topright', c("Train error", "Test error"), lty = c(1,1), lwd = c(2.5, 2.5),
       col = c("red", "blue"))
```