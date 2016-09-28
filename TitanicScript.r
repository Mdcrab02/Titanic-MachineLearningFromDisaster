#Start off by loading up some of the libraries that might be useful
#Manipulation
library(tidyr)
library(dplyr)
#Modeling
library(rpart)
library(randomForest)
library(rattle)
library(car)
library(caret)
library(e1071)
library(C50)
library(RWeka)
#Visualization
library(rpart.plot)
library(RColorBrewer)
library(ggplot2)
library(reshape2)

#import the train and test datasets from the directory

#import the test data from the file name test_orig
newtest <- test_orig
#insert a blank column to hold future values for Survived
newtest$Survived <- ""
#combine test and train into one dataset
titanicData <- rbind(train_orig,newtest)
#check out what the total dataset looks like now
str(titanicData)
View(titanicData)
#refactor the features Survived and Pclass into factors
titanicData$Survived <- as.factor(titanicData$Survived)
titanicData$Pclass <- as.factor(titanicData$Pclass)
#verify the changes
str(titanicData)

#Add some new features to the dataset for future use

#the first is to create a new feature called family_size to account for the family size of each passenger
  #wherein the famil is the number of siblings and spouses, parents and children, and the individual
titanicData$family_size <- titanicData$SibSp + titanicData$Parch + 1

#Next, add a feature to account for the passengers title.  There are several passengers that are the only
  #ones with individual titles.  This presents a problem for classification algorithms.
titanicData$Title <- sub(pattern = ".*,\\s(.*\\.)\\s.*", replacement = "\\1", x = titanicData$Name)
titanicData$Title[titanicData$Title %in% c("Don.", "Dona.", "the Countess.")] <- "Lady." 
titanicData$Title[titanicData$Title %in% c("Ms.", "Mlle.")] <- "Miss." 
titanicData$Title[titanicData$Title %in% c("Mme.", "Mrs. Martin (Elizabeth L.")] <- "Mrs." 
titanicData$Title[titanicData$Title %in% c("Jonkheer.", "Don.")] <- "Sir." 
titanicData$Title[titanicData$Title %in% c("Col.", "Capt.", "Major.")] <- "Officer." 
titanicData$Title <- factor(titanicData$Title)

ggplot(titanicData[1:891,], aes(x=family_size, fill=Survived)) +
  geom_histogram(binwidth = 1) +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass, Title") +
  xlab("family size") +
  ylab("Total count") +
  ylim(0,300) +
  labs(fill = "Survived")

#Check for NA values in each feature of the dataset
table(is.na(titanicData$PassengerId))
table(is.na(titanicData$Survived))
table(is.na(titanicData$Pclass))
table(is.na(titanicData$Name))
table(is.na(titanicData$Sex))
table(is.na(titanicData$Age)) #NAs found
table(is.na(titanicData$SibSp))
table(is.na(titanicData$Parch))
table(is.na(titanicData$Ticket))
table(is.na(titanicData$Fare)) #NAs found
table(is.na(titanicData$Cabin))
table(is.na(titanicData$Embarked))
table(is.na(titanicData$family_size))
table(is.na(titanicData$Title))

#So Age has some NA values in the feature vector.  One cannot exactly not have an age, so let's impute
  #these missing values by predicting them based on values we know
predicted_age <- rpart(Age ~ Pclass 
                       + Sex 
                       + SibSp 
                       + Parch 
                       + Fare 
                       + Embarked 
                       + Title #new
                       + family_size #new
                       , data = titanicData[!is.na(titanicData$Age),]
                       , method = "anova")
titanicData$Age[is.na(titanicData$Age)] <- predict(predicted_age, titanicData[is.na(titanicData$Age),])

#Fare also has some NA values in the feature vector.  This is not helpful for solving the problem, so
  #impute these missing variables as well using the median value (not the average) of the non-normal distribution
titanicData$Fare[is.na(titanicData$Fare)==TRUE] <- median(titanicData$Fare,na.rm=TRUE)

#In the dataset, two passengers do not have an embarked location.  They had to come from somewhere, assuming
  #they did not spawn from the ocean itself.  Because most passengers embarked from "S," let's say these two
  #did as well
titanicData$Embarked[c(62, 830)] <- "S"
titanicData$Embarked <- factor(titanicData$Embarked)

#Now add some new features to practice feature engineering and to see what may or may not be helpful
  #These are all be binomials that might help us with our classification
titanicData$Child[titanicData$Age>=18] <- 0
titanicData$Child[titanicData$Age<18] <- 1

titanicData$Elder[titanicData$Age>=55] <- 1
titanicData$Elder[titanicData$Age<55] <- 0

titanicData$HighFare[titanicData$Fare>=200] <- 1
titanicData$HighFare[titanicData$Fare<200] <- 0

titanicData$LowFare[titanicData$Fare<=8] <- 1
titanicData$LowFare[titanicData$Fare>8] <- 0

#The group of features below are categorical in nature and polynomial
titanicData$TicketAB<-substring(titanicData$Ticket,1,1)
titanicData$TicketAB<-factor(titanicData$TicketAB)

titanicData$FamCat[titanicData$family_size == 1] <- 0
titanicData$FamCat[titanicData$family_size < 5 & titanicData$family_size > 1] <- 1
titanicData$FamCat[titanicData$family_size > 4] <- 2

titanicData$AgeGrp[titanicData$Age <= 10] <- 1
titanicData$AgeGrp[titanicData$Age > 10 & titanicData$Age <= 19] <- 2
titanicData$AgeGrp[titanicData$Age > 19 & titanicData$Age <=50] <- 3
titanicData$AgeGrp[titanicData$Age > 50] <- 4

titanicData$FareGrp[titanicData$Fare <=10] <- 1
titanicData$FareGrp[titanicData$Fare >10 & titanicData$Fare <=50] <- 2
titanicData$FareGrp[titanicData$Fare >50 & titanicData$Fare <=100] <- 3
titanicData$FareGrp[titanicData$Fare >100] <- 4

#Look at the Cabin feature
str(titanicData$Cabin)
table(titanicData$Cabin)
#This beast has 187 levels, which can lower the performance of classification algorithms
titanicData$Cabin <- as.character(titanicData$Cabin)
titanicData$Cabin[is.na(titanicData$Cabin)] <- "U"
titanicData$Cabin[titanicData$Cabin==""] <- "U"
titanicData$Cab<-substring(titanicData$Cabin,1,1)
titanicData$Cab<-factor(titanicData$Cab)
titanicData$Cabin<-factor(titanicData$Cabin)
#Our new feature, Cab, has only 9 levels and still represents most of the information from
  #the Cabin feature vector
str(titanicData$Cab)

#Add some categorical features to group families into types by their size
titanicData$FamType[titanicData$family_size == 1] <- 'single'
titanicData$FamType[titanicData$family_size < 5 & titanicData$family_size > 1] <- 'small'
titanicData$FamType[titanicData$family_size > 4] <- 'large'
titanicData$FamType<-factor(titanicData$FamType)

#Some passengers are actually registered to multiple cabins.  This can be related to a passengers
  #socioeconomic status which can affect survivability
titanicData$MultiCab <- as.factor(ifelse(str_detect(titanicData$Cabin, " "), "Y", "N"))

#The features for Fare and Age have values that are outliers and values that disturb the distribution
  #so create two new features that are normalized representations of both
  #probably will not use them, but making them anyway for now
titanicData$FareNorm <- data.Normalization(titanicData$Fare, type="n1",normalization="column")
titanicData$AgeNorm <- data.Normalization(titanicData$Age, type="n12",normalization="column")

#Check on the structure of the dataset now
str(titanicData)
#Refactor some of the new features
titanicData$Child<-factor(titanicData$Child)
titanicData$Elder<-factor(titanicData$Elder)
titanicData$HighFare<-factor(titanicData$HighFare)
titanicData$LowFare<-factor(titanicData$LowFare)
titanicData$FamCat<-factor(titanicData$FamCat)
titanicData$AgeGrp<-factor(titanicData$AgeGrp)
titanicData$FareGrp<-factor(titanicData$FareGrp)
#Transform some of the others from num to int
titanicData$Age <- as.integer(titanicData$Age)
titanicData$family_size <- as.integer(titanicData$family_size)

#Now split the data back into training and test sets
train <- titanicData[1:891,]
train$Survived <- factor(train$Survived)
str(train)

test <- titanicData[892:1309,]
test$Survived <- factor(test$Survived)
str(test)

#Set seed for reproducibility
set.seed(1)

#First trying the random forest algorithm with cross validation
estPerformance <- c(0,0,0,0,0,0,0,0,0,0)
for(h in 1:10){
  #For each of the ten
  accs<-c(0,0,0,0)
  n <- nrow(train)
  shuffled <- train[sample(n),]
  for (i in 1:4) {
    indices <- (((i-1) * round((1/4)*nrow(shuffled))) + 1):((i*round((1/4) * nrow(shuffled))))
    
    # Exclude them from the train set
    train2 <- shuffled[-indices,]
    
    # Include them in the test set
    test2 <- shuffled[indices,]
    
    m <- randomForest(as.factor(Survived) ~ Title + 
                        Sex + #
                        Age + #
                        TicketAB + #
                        Pclass+
                        family_size+
                        FareGrp #+ #
                        ,ntree=1500, data=train,importance=TRUE)
    
    #Make a prediction on the test set using tree
    pred <- predict(m, test2, type="class")
    
    #Assign the confusion matrix to conf
    conf<-table(test2$Survived,pred)
    
    #Assign the accuracy of this model to the ith index in accs
    accs[i]<-sum(diag(conf))/sum(conf)        
  }
  estPerformance[h] <- mean(accs)
}
#Check out the confusion matrix
conf
#Have a look at the different features used and their performance power
varImpPlot(m)
#Check out the average of all of the folded results using this model and ensuring the records
#are shuffled prior to each evaluation
mean(estPerformance)
#The random forest performance averages around .892
#Create predictions based on the model above
myPrediction <- predict(m, test)
#Generate the solution file for the competition
mySolution <- data.frame(PassengerID=test$PassengerId, Survived=myPrediction)
write.csv(mySolution,file="my_solution(rForest).csv", row.names=FALSE)



#Now trying a J48 classification tree from Weka
m <- J48(as.factor(Survived) ~ Title
         +Sex
         +Age
         +TicketAB
         +Pclass
         +family_size
         +FareGrp
         ,data=train)
#Use the built-in evaluation methods to analyze the performance of the J48
e <- evaluate_Weka_classifier(m,cost = matrix(c(0,2,1,0), ncol = 2),
                              numFolds = 10, complexity = TRUE, #newdata=test,
                              seed = 123, class = TRUE)
#View the summary of the performance results
e$details
#Even though it is still looking at the training set, the performance is around 83%, which is a
  #pretty good sign

#Now use some cross validation to further test the performance of the J48 model
estPerformance <- c(0,0,0,0,0,0,0,0,0,0)
for(h in 1:10){
  #For each of the ten
  accs<-c(0,0,0,0)
  n <- nrow(train)
  shuffled <- train[sample(n),]
  for (i in 1:4) {
    indices <- (((i-1) * round((1/4)*nrow(shuffled))) + 1):((i*round((1/4) * nrow(shuffled))))
    
    # Exclude them from the train set
    train2 <- shuffled[-indices,]
    
    # Include them in the test set
    test2 <- shuffled[indices,]
    
    m <- J48(as.factor(Survived) ~ Title
             +Age
             +TicketAB
             +FamType
             +Fare
             +Sex
             +Pclass
             +Cab
             +FamCat
             +family_size
             +MultiCab
             ,train)
    
    #Make a prediction on the test set using tree
    pred <- predict(m, test2, type="class")
    
    #Assign the confusion matrix to conf
    conf<-table(test2$Survived,pred)
    
    #Assign the accuracy of this model to the ith index in accs
    accs[i]<-sum(diag(conf))/sum(conf)        
  }
  estPerformance[h] <- mean(accs)
}
#Check out the confusion matrix
conf
#Check out the average of all of the folded results using this model and ensuring the records
  #are shuffled prior to each evaluation
mean(estPerformance)
#Performance average hovers around .866, which seems good enough to make a submission to the contest
#Use the built-in evaluation methods to analyze the performance of the J48
e <- evaluate_Weka_classifier(m,cost = matrix(c(0,2,1,0), ncol = 2),
                              numFolds = 10, complexity = TRUE,
                              seed = 123, class = TRUE)
#View the summary of the performance results
e$details
#The built-in performance measurement gives the model around .829, which is still okay
#Create predictions based on the model above
myPrediction <- predict(m, test)
#Generate the solution file
mySolution <- data.frame(PassengerID=test$PassengerId, Survived=myPrediction)
write.csv(my_solution,file="my_solution(J48).csv", row.names=FALSE)