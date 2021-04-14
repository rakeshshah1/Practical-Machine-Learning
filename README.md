---
title: "Practical Machine Learning"
author: "Eldrick Fonollera"
date: "December 10, 2018"
output: html_document
---

## INTRODUCTION

### This is the project of practical machine learning. Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

## DOWNLOAD THE FILES

```{r}
library(caret)
```

## LOADING DATA

```{r}
trainingDataSet<- read.csv("pml-training.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
testingDataSet<- read.csv("pml-testing.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
dim(trainingDataSet)
```

```{r}
dim(testingDataSet)
```

## DATA CLEANING

```{r}
trainingDataSet <- trainingDataSet[,(colSums(is.na(trainingDataSet)) == 0)]
dim(trainingDataSet)
```

```{r}
testingDataSet <- testingDataSet[,(colSums(is.na(testingDataSet)) == 0)]
dim(testingDataSet)
```

## PROCESS DATA

```{r}
numericalsIdx <- which(lapply(trainingDataSet, class) %in% "numeric")

preprocessModel <-preProcess(trainingDataSet[,numericalsIdx],method=c('knnImpute', 'center', 'scale'))
pre_trainingDataSet <- predict(preprocessModel, trainingDataSet[,numericalsIdx])
pre_trainingDataSet$classe <- trainingDataSet$classe

pre_testingDataSet <-predict(preprocessModel,testingDataSet[,numericalsIdx])
```

## REMOVE ZERO VARIABLES

```{r}
nzv <- nearZeroVar(pre_trainingDataSet,saveMetrics=TRUE)
pre_trainingDataSet <- pre_trainingDataSet[,nzv$nzv==FALSE]

nzv <- nearZeroVar(pre_testingDataSet,saveMetrics=TRUE)
pre_testingDataSet <- pre_testingDataSet[,nzv$nzv==FALSE]
```

## INVESTIGATE

```{r}
set.seed(12031987)
idxTrain<- createDataPartition(pre_trainingDataSet$classe, p=3/4, list=FALSE)
training<- pre_trainingDataSet[idxTrain, ]
validation <- pre_trainingDataSet[-idxTrain, ]
dim(training) ; dim(validation)
```

## MODEL TRAINING

```{r}
library(randomForest)
modFitrf <- train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE, importance=TRUE )
modFitrf
```

## CROSS VALIDATION

```{r}
predValidRF <- predict(modFitrf, validation)
confus <- confusionMatrix(validation$classe, predValidRF)
confus$table
```

## ACCURACY

```{r}
accur <- postResample(validation$classe, predValidRF)
modAccuracy <- accur[[1]]
modAccuracy
```

## OUT OF SAMPLE ERROR

```{r}
out_of_sample_error <- 1 - modAccuracy
out_of_sample_error
```


## DECISION

As the accuracy of the model is good (~ 100%) and the out-of-sample error is low, I have decided to utilise the Random Forest algorithm as the chosen model for the project. The pre-processing and cross-validation methods were applied effectively to improve the accuracy of the predictions.

## APPLYING THE MODEL

```{r}
pred_final <- predict(modFitrf, pre_testingDataSet)
pred_final
```
