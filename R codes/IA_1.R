library(dplyr)
library(caret)
library(data.table)
library(ggplot2)
library(bestglm)
library(e1071)
library(grid)
library(gridExtra)
library(ff)
library(pROC)
library(tidyr)
library(ggthemes)
library(ROCR)
library(gbm)
library(randomForest)
ds <- read.csv(file = 'train.csv', nrows = 100000)
nobs <- nrow(ds)
train <- sample(nobs, 0.7*nobs)
test <- setdiff(seq_len(nobs), train)
ds[] <- lapply(ds, factor)
target <- "click"
form <- formula(paste(target, "~ ."))


##########################
##Exploratory Data Analysis
##########################
library(plyr)
library(sjPlot)
Data=data.table(ds) 

Reason_banner_pos=Data[, count(click), by = banner_pos]

sjp.xtab(Data$click, 
         Data$banner_pos,
         bar.pos = c("dodge"),
         show.total = FALSE)
sjp.xtab(Data$click, 
         Data$site_category,
         bar.pos = c("dodge"),
         show.total = FALSE)
sjp.xtab(Data$click, 
         Data$app_category,
         bar.pos = c("dodge"),
         show.total = FALSE)
sjp.xtab(Data$click, 
         Data$device_type,
         bar.pos = c("dodge"),
         show.total = FALSE)
sjp.xtab(Data$click, 
         Data$C16,
         bar.pos = c("dodge"),
         show.total = FALSE)
sjp.xtab(Data$click, 
         Data$C18,
         bar.pos = c("dodge"),
         show.total = FALSE)
sjp.xtab(Data$click, 
         Data$C19,
         bar.pos = c("dodge"),
         show.total = FALSE)


subset_data_for_importance <- ds[1:5000, ]
subset_data_for_importance[] <- lapply(subset_data_for_importance, factor) 
nobs_subset_gbm <- nrow(subset_data_for_importance)
train_importance <- sample(nobs_subset_gbm, 0.7*nobs_subset_gbm)
data_importance <- subset_data_for_importance[train_importance,-which(names(ds) %in% c("id","hour","device_ip"))]

GBM_model = gbm(
  form,data = data_importance,
  n.trees = 10000, 
  distribution = "gaussian",
  cv.folds=2
)
summary(GBM_model)



subsample <- ds[1:10000,]
subsample[] <- lapply(subsample, factor) 
nobs_subsample <- nrow(subsample)
train_subsample <- sample(nobs_subsample, 0.7 * nobs_subsample)
test_subsample <- setdiff(seq_len(nobs_subsample), train_subsample)
train_subsample_data <- subsample[train_subsample,c("click","device_model","site_id","C14","device_id","site_domain","app_id")] #app_id extra
test_subsample_data <- subsample[test_subsample,c("click","device_model","site_id","C14","device_id","site_domain","app_id")]
cost_fp <- 100
cost_fn <- 200

###################
#SVM
##################
train_data_svm <- train_subsample_data
test_data_svm <- test_subsample_data
tuned_svm <- tune.svm(form, data=train_data_svm, gamma = 10^(-7:-5), cost = 10^(1:3))
summary(tuned_svm)

#ds_y <- ds[test,target]
#ds_x <- ds[test,-1]
svmmodel <- svm(form, data = train_data_svm, gamma = 10^(-7), cost = 10, probability = TRUE )
svmPred_prob_test <- predict(svmmodel, newdata= test_data_svm, probability = TRUE)
svmPred_prob_train <- predict(svmmodel, train_data_svm, probability = TRUE)
train_data_svm$prediction <- attr(svmPred_prob_train,"probabilities")[,2]
test_data_svm$prediction <- attr(svmPred_prob_test,"probabilities")[,2]
test_data_svm$prediction2 <- attr(svmPred_prob_test,"probabilities")[,1]
#accuracy_info_svm <- AccuracyCutoffInfo( train = train_data_svm, test = test_data_svm, predict = "prediction", actual = "click" )
#accuracy_info_svm$plot
cm_info_svm <- ConfusionMatrixInfo( data = test_data_svm, predict = "prediction", actual = "click", cutoff = .5 )
cm_info_svm$plot
cm_info_svm$data
roc_info_svm <- ROCInfo( data = cm_info_svm$data, predict = "predict", actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )
grid.draw(roc_info_svm$plot)
test_data_svm$predicted_click <- ifelse(test_data_svm$prediction>0.85,"1","0") ## cutoff decided from plot
#table(test_data_svm$predicted_click, test_data_svm$click)
confusionMatrix_svm <- confusionMatrix(test_data_svm$predicted_click, test_data_svm$click)
mtx.svm[,,1] <- rbind(confusionMatrix_svm$overall,confusionMatrix_svm$byClass[1:7])
test_data_svm$click <- as.integer(test_data_svm$click)
test_data_svm[test_data_svm$click==1,c('click')] <- 0
test_data_svm[test_data_svm$click==2,c('click')] <- 1
logloss_svm <- LogLoss(test_data_svm$click,test_data_svm$prediction)
#0.45

###############################
#K-Nearest NEighbors
##############################
library(RWeka)
train_data_knn <- ds[train,c("click","device_model","site_id","C14","device_id","site_domain","app_id")]
test_data_knn <- ds[test,c("click","device_model","site_id","C14","device_id","site_domain","app_id")]
classifier_knn <- IBk(form, data = train_data_knn, control = Weka_control(K = 20, X = TRUE))
knn_evaluation_result <- evaluate_Weka_classifier(classifier_knn, newdata = test_data_knn, numFolds = 3)
classifier #using 16 neighbors
knn_prediction <- predict(classifier_knn, test_data_knn, type = "probability")
knn_prediction_click <- predict(classifier_knn, test_data_knn)
#test_data_knn$prediction <- knn_prediction[,1]
#cm_info_knn <- ConfusionMatrixInfo( data = test_data_knn, predict = "prediction", actual = "click", cutoff = 0.5 )
#cm_info_knn$plot
#cm_info_knn$data
#roc_info_knn <- ROCInfo( data = cm_info_knn$data, predict = "predict", actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )
#grid.draw(roc_info_knn$plot)
#test_data_knn$predicted_click <- ifelse(test_data_rf$prediction>0.41,"1","0")
confusionMatrix_knn <- confusionMatrix(knn_prediction_click, test_data_knn$click)
#confusionMatrix_knn1 <- confusionMatrix(test_data_knn$predicted_click, test_data_knn$click)
test_data_knn$click <- as.integer(test_data_knn$click)
test_data_knn[test_data_knn$click==1,c('click')] <- 0
test_data_knn[test_data_knn$click==2,c('click')] <- 1
logloss_knn <- LogLoss(test_data_knn$click, knn_prediction)
## 1.43
mtx.knn[,,1] <- rbind(confusionMatrix_knn$overall,confusionMatrix_knn$byClass[1:7])
###############################
#Random Forest
################################
#opt <- trainControl(method='repeatedcv', number=10, repeats=15, classProbs=TRUE)
#fit.rf <- train(form, data=learn, method='rf',tuneGrid=expand.grid(.mtry=1:6),ntree=1000,metric='Accuracy', trControl=opt)

train_data_rf <- subsample[train_subsample,-which(names(ds) %in% c("id","hour","device_ip","site_id","site_domain","app_id","device_id","device_ip","device_model","C14","C17","C20"))]
test_data_rf <- subsample[test_subsample,-which(names(ds) %in% c("id","hour","device_ip","site_id","site_domain","app_id","device_id","device_ip","device_model","C14","C17","C20"))]
train_data_rf$prediction = NULL
rf_fit <- randomForest(form, data = train_data_rf, ntree = 2000, importance = TRUE, type = "prob")
importance(rf_fit)
varImpPlot (rf_fit)
rf_prediction_test = predict(rf_fit,newdata = test_data_rf, type = "prob")
rf_prediction_train = predict(rf_fit,newdata = train_data_rf, type = "prob")
train_data_rf$prediction <- rf_prediction_train[,2]
test_data_rf$prediction <- rf_prediction_test[,2]
accuracy_info_rf <- AccuracyCutoffInfo( train = train_data_rf, test = test_data_rf, predict = "prediction", actual = "click" )
accuracy_info_rf$plot
cm_info_rf <- ConfusionMatrixInfo( data = test_data_rf, predict = "prediction", actual = "click", cutoff = 0.1 )
cm_info_rf$plot
cm_info_rf$data
roc_info_rf <- ROCInfo( data = cm_info_rf$data, predict = "predict", actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )
grid.draw(roc_info_rf$plot)
test_data_rf$predicted_click <- ifelse(test_data_rf$prediction>0.1805,"1","0")
confusionMatrix_rf <- confusionMatrix(test_data_rf$predicted_click, test_data_rf$click)
test_data_rf$click <- as.integer(test_data_rf$click)
test_data_rf[test_data_rf$click==1,c('click')] <- 0
test_data_rf[test_data_rf$click==2,c('click')] <- 1
logloss_rf <- LogLoss(test_data_rf$click,test_data_rf$prediction)
#0.673
CM.rf <- confusionMatrix(test_data_rf$predicted_click, test_data_rf$click)
mtx.rf[,,1] <- rbind(confusionMatrix_rf$overall,confusionMatrix_rf$byClass[1:7])



################################
#NAive Bayes
###############################
data_NB <- read.csv(file = 'train.csv', nrows = 1000000)
nobs_NB <- nrow(data_NB)
train_NB <- sample(nobs_NB, 0.7*nobs_NB)
test_NB <- setdiff(seq_len(nobs_NB), train_NB)
data_NB[] <- lapply(data_NB, factor)
train_data_NB <- data_NB[train_NB,c("click","device_model","site_id","C14","device_id","site_domain","app_id")]
test_data_NB <- data_NB[test_NB,c("click","device_model","site_id","C14","device_id","site_domain","app_id")]
classifier_NB <- naiveBayes(form, data=train_data_NB)
predicted_nb_test <- predict(classifier_NB, test_data_NB, type = "raw")
test_data_NB$prediction <- predicted_nb_test[,2]
predicted_nb_class <- predict(classifier_NB, test_data_NB, type = "class")
confusionMatrix_NB1 <- confusionMatrix(predicted_nb_class, test_data_NB$click)
test_data_NB$click <- as.integer(test_data_NB$click)
test_data_NB[test_data_NB$click==1,c('click')] <- 0
test_data_NB[test_data_NB$click==2,c('click')] <- 1
logloss_NB <- LogLoss(test_data_NB$click,test_data_NB$prediction)
#0.675
mtx.nb[,,1] <- rbind(confusionMatrix_NB1$overall,confusionMatrix_NB1$byClass[1:7])
#####################################
#Best Susbset Selection: Logistic Regression
#####################################

ignore <- c(
  "id",
  "hour",
  "site_id",
  "site_domain",
  "app_id",
  "device_id",
  "device_ip",
  "device_model",
  "C14",
  "C17",
  "C20",
  "C21"
)
vars <- setdiff(names(ds), ignore)
inputs <- setdiff(vars, target)
form <- formula(paste(target, "~ ."))
actual <- ds[test, target]
train_data_bestsubset <- ds[train,]
test_data_bestsubset <- ds[test,]
library(bestglm)

train_data_bestsubset <- within(train_data_bestsubset, {
  y    <- click 
  click  <- NULL
})
#train_data <-train_data[, c("C1","banner_pos","site_category","app_domain","app_category","device_type","device_conn_type","C15","C16","C18","C19","Count","y")]
bestlogistic_subset1 <-
  bestglm(Xy = train_data_bestsubset,
          family = binomial,          # binomial family for logistic
          IC = "AIC",                 # Information criteria for
          method = "exhaustive")


test_data <- test_data[test_data$site_category!="bcf865d9",]
test_data <- test_data[test_data$site_category!="110ab22d",]
test_data <- test_data[test_data$C19!="1071",]
logistic_prediction <- predict(bestlogistic_subset$BestModel, test_data, type="response")
logistic_prediction_clicks <- round(predict(bestlogistic_subset$BestModel, test_data, type="response"))
confusionMatrix(logistic_prediction_clicks, test_data$click)
table(test_data$click,logistic_prediction_clicks)

train_data$prediction <- predict(bestlogistic_subset$BestModel, newdata = train_data, type = "response" )
test_data$prediction  <- predict(bestlogistic_subset$BestModel, newdata = test_data , type = "response" )
library(ggthemes)
train_data <- within(train_data, {
  click    <- y 
  y  <- NULL
})
ggplot( train_data, aes( prediction, color = as.factor(click) ) ) + 
  geom_density( size = 1 ) +
  ggtitle( "Training Set's Predicted Score" ) + 
    scale_color_economist( name = "data", labels = c( "non-click", "click" ) ) + 
  theme_economist()


ggthemr("light")
accuracy_info$plot



cm_info <- ConfusionMatrixInfo( data = data, predict = "prediction", 
                                actual = "click", cutoff = .2 )
cm_info$plot
print(cm_info$data)
library(ggthemr)





cost_fp <- 100
cost_fn <- 300
roc_info <- ROCInfo( data = cm_info$data, predict = "predict", 
                     actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )
grid.draw(roc_info$plot)


cm_info <- ConfusionMatrixInfo( data = test_data, predict = "prediction", 
                                actual = "click", cutoff = roc_info$cutoff )
cm_info$plot
Likelycustomers <- test_data[ test_data$prediction >= roc_info$cutoff, ]
list( head(Likelycustomers), nrow(Likelycustomers) )
table(Likelycustomers$click)
formula(bestlogistic_subset$BestModel)
test_data$predicted_trshold <- ifelse(test_data$prediction>0.18,"1","0")
confusionMatrix(test_data$predicted_trshold,test_data$click)
y ~ banner_pos + site_category + app_category + device_type + C16 + C18 + C19

bestlogistic_subset$BestModels
summary(bestlogistic_subset$BestModel)

#############
# Model Performance
#############

mtx.mlogit <- array(0,c(4,7,no.resamp))
mtx.nb <- mtx.mlogit
mtx.knn <- mtx.mlogit
mtx.rf <- mtx.mlogit
mtx.svm <- mtx.mlogit
mtx.ftrl <- mtx.mlogit

library(ggplot2)
grp1 <- c(rep(1,dim(mtx.rf)[3]),rep(2,dim(mtx.svm)[3]),
          rep(3,dim(mtx.knn)[3]),rep(4,dim(mtx.nb)[3]),
          rep(5,dim(mtx.mlogit)[3]))
grp1 <- c(grp1,grp1)
grp1 <- factor(grp1,labels=c("RF","SVM","KNN","NBayes","Logistic"))
accuracy <- c(nb=mtx.rf[1,1,],mlogit=mtx.svm[1,1,],
              knn=mtx.knn[1,1,],rf=mtx.nb[1,1,],svm=mtx.mlogit[1,1,])
kappa <- c(nb=mtx.rf[1,2,],mlogit=mtx.svm[1,2,],
           knn=mtx.knn[1,2,],rf=mtx.nb[1,2,],svm=mtx.mlogit[1,2,])
grp2 <- factor(c(t(matrix(rep(1:2,length(kappa)),2))),
               labels=c("Accuracy","Kappa"))

dtset1 <- data.frame(grp1,grp2,perf=c(accuracy,kappa))
p1 <- ggplot(aes(y=perf,x=grp1),data=dtset1) + 
  geom_boxplot(aes(fill=grp1)) + 
  coord_flip() + facet_wrap(~grp2, ncol=1, scales="fixed")+
  scale_fill_discrete(guide=F) +
  scale_x_discrete(name="")+scale_y_continuous(name="")+
  theme(text=element_text(size = 24))

postscript(file="fig7A.eps", height=8, width=8, horizontal= F, 
           paper="special", colormodel="rgb")
print(p1)
dev.off()


grp1 <- c(rep(1,dim(mtx.rf)[3]),rep(2,dim(mtx.nnet)[3]),
          rep(3,dim(mtx.knn)[3]),rep(4,dim(mtx.nb)[3]),
          rep(5,dim(mtx.mlogit)[3]))
grp1 <- c(grp1,grp1)
grp1 <- factor(grp1,labels=c("RF","SVM","KNN","NBayes","MLogit"))
sens_nonclcik <- c(nb=mtx.rf[2,1,],mlogit=mtx.nnet[2,1,],
                   knn=mtx.knn[2,1,],rf=mtx.nb[2,1,],nnet=mtx.mlogit[2,1,])
sens_click <- c(nb=mtx.rf[4,1,],mlogit=mtx.nnet[4,1,],
                knn=mtx.knn[4,1,],rf=mtx.nb[4,1,],nnet=mtx.mlogit[4,1,])
grp2 <- factor(c(t(matrix(rep(1:2,length(sens_nonclcik)),2))),
               labels=c("Sensitivity non click","Sensitivity click"))
dtset2 <- data.frame(grp1,grp2,perf=c(sens_nonclcik,sens_click))
p2 <- ggplot(aes(y=perf,x=grp1),data=dtset2) + 
  geom_boxplot(aes(fill=grp1)) + 
  coord_flip() + facet_wrap(~grp2, ncol=1, scales="fixed")+
  scale_fill_discrete(guide=F) +
  scale_x_discrete(name="")+scale_y_continuous(name="")+
  theme(text=element_text(size = 24))
print(p2)



library(raster)
dtset <- rbind(dtset1,dtset2)

mtx.mn <- round(matrix(by(dtset$perf,list(dtset$grp1,dtset$grp2),mean),5),2)
mtx.sd <- round(matrix(by(dtset$perf,list(dtset$grp1,dtset$grp2),sd),5),2)
mtx.cv <- round(matrix(by(dtset$perf,list(dtset$grp1,dtset$grp2),cv),5),1)
tab4 <- matrix(c(rbind(mtx.mn,mtx.sd,mtx.cv)),5)
rownames(tab4) <- c("RF","SVM","KNN","NBayes","MLogit")
colnames(tab4) <- c("Ave.Acc","SD.Acc","CV.Acc","Ave.Kap","SD.Kap","CV.Kap",
                    "Ave.SensW","SD.SensW","CV.SensW","Ave.SensL","SD.SensL","CV.SensL",
                    "Ave.SensD","SD.SensD","CV.SensD")
print(tab4)




###########################
#Regularized Logistic Regression
##########################
library(glmnet)
set.seed(999)
ignore <- c(
  "id",
  "hour",
  "device_ip"
)
vars <- setdiff(names(ds), ignore)
train_data_regularized <- subsample[train_subsample,vars]
test_data_regularized <- subsample[test_subsample,vars]
x <- model.matrix( ~ .-1, train_data_regularized[,-1])
x_test <- model.matrix(click~.,test_data_regularized)
y <- as.double(as.matrix(train_data_regularized[, 1])) # Only class
cv.lasso <- cv.glmnet(x, y = factor(train_data_regularized$click), family='binomial', alpha=1, standardize=TRUE, type.measure='auc')
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
bestlam <- cv.lasso$lambda.min
lasso_modfit <- glmnet(x, y = factor(train_data_regularized$click), alpha = 1, family = "binomial", lambda = bestlam, standardize = TRUE)
lasso_prob_test <- predict(cv.lasso,newx=x_test,s=bestlam,type="response")
lasso_prob_train <- predict(cv.lasso,newx=x,s=bestlam,type="response")
train_data_regularized$prediction <- lasso_prob_train
test_data_regularized$prediction <- lasso_prob
lasso_prediction <- predict(cv.lasso,newx=x_test,s=bestlam,type="class")
confusionMatrix(lasso_prediction, test_data_regularized$click)
cm_info_regularized <- ConfusionMatrixInfo(data = test_data_regularized, predict = "prediction",actual = "click", cutoff = .3 )
cm_info_regularized$plot
cm_info_regularized <- ConfusionMatrixInfo(data = train_data_regularized, predict = "prediction",actual = "click", cutoff = .3 )

roc_info_regularized <- ROCInfo(data=cm_info_regularized$data, predict = "predict.1",actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )
grid.draw(roc_info_regularized$plot)
test_data_regularized$predictedclick <- ifelse(test_data_regularized$prediction>0.2689739,"1","0")
confusionMatrix_RLR <- confusionMatrix(test_data_regularized$predictedclick, test_data_regularized$click)
test_data_regularized$click <- as.integer(test_data_regularized$click)
test_data_regularized[test_data_regularized$click==1,c('click')] <- 0
test_data_regularized[test_data_regularized$click==2,c('click')] <- 1
logloss_Regularized <- LogLoss(test_data_regularized$click,test_data_regularized$prediction)
mtx.mlogit[,,1] <- rbind(confusionMatrix_RLR$overall,confusionMatrix_RLR$byClass[1:7])

#################
# FTRL
#################
data_FTRL <- read.csv(file = 'train_subset.csv', nrows = 100000)
cm_info_FTRL <- ConfusionMatrixInfo( data = data_FTRL, predict = "click.1",actual = "click", cutoff = .3 )
cm_info_FTRL$plot
roc_info_FTRL <- ROCInfo( data = cm_info_FTRL$data, predict = "predict",actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )
grid.draw(roc_info_FTRL$plot)
data_FTRL$predictedclick <- ifelse(data_FTRL$click.1>0.293,"1","0")
confusionMatrix_FTRL <- confusionMatrix(data_FTRL$predictedclick, data_FTRL$click)
logloss_FTRL <- LogLoss(data_FTRL$click,data_FTRL$click.1)
# 0.4346
mtx.ftrl[,,1] <- rbind(confusionMatrix_FTRL$overall,confusionMatrix_FTRL$byClass[1:7])

