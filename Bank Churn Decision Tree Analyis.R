# BANK CHURN DECISION TREE ANALYSIS############################################

# Load/install packages
install.packages("rpart.plot")
install.packages("Hmisc")
install.packages("ggpubr")
install.packages('caret', dependencies = TRUE)
install.packages('e1071', dependencies=TRUE)
install.packages("pdp")
install.packages("ggplot2")
install.packages("gridExtra")

library(rpart)
library(rpart.plot)
library(caret)
library(Hmisc)
library(ggpubr)
library(pdp)
library(ggplot2)
library(gridExtra)


# Load the dataset
full_data <- read.csv("F:/HW University/Big Data Course/data/Bank customers churn.csv")

# Check the dataset structure using the str() function
str(full_data)

#Describing the dataset
install.packages("psych")
library(psych)
describe(full_data)

# convert the response into two level factor
full_data$Exited=as.factor(full_data$Exited)
#plot
n1 <- ggplot(data = full_data, aes(CreditScore, color = Exited))+geom_freqpoly(binwidth = 5, size
                                                                               = 1)
n2 <- ggplot(data = full_data, aes(Age, color = Exited))+geom_freqpoly(binwidth = 5, size = 1)
n3 <- ggplot(data = full_data, aes(Balance, color = Exited))+geom_freqpoly(binwidth = 10000, size
                                                                           = 1)
n4 <- ggplot(data = full_data, aes(EstimatedSalary, color = Exited))+geom_freqpoly(binwidth =
                                                                                     10000, size = 1)

# Chart the numerical variables on one canvas
ggarrange(n1,n2,n3,n4,ncol = 2, nrow = 2)

# Visualise other variables
c1 <- ggplot(data = full_data, aes(Gender, fill = Exited))+geom_bar(position = "fill", stat = "count")
c2 <- ggplot(data = full_data, aes(Geography, fill = Exited))+geom_bar(position = "fill", stat = "count")
c3 <- ggplot(data = full_data, aes(Tenure, fill = Exited))+geom_bar(position = "fill", stat = "count")
c4 <- ggplot(data = full_data, aes(NumOfProducts, fill = Exited))+geom_bar(position = "fill", stat = "count")
c5 <- ggplot(data = full_data, aes(IsActiveMember, fill = Exited))+geom_bar(position = "fill", stat = "count")
c6 <- ggplot(data = full_data, aes(HasCrCard, fill = Exited))+geom_bar(position = "fill", stat = "count")
# Chart the variables on 2x2 canvases
ggarrange(c1,c2,c3,c4, ncol = 2, nrow = 2 )
# Chart the variables on 2x2 canvases
ggarrange(c5,c6, ncol = 2, nrow = 1 )

# Remove CustomerId, Surname, RowNumber, and HasCrCard from
# the dataset as these variables are not needed for modelling
model_data = subset(full_data, select = -c(CustomerId, Surname, RowNumber, HasCrCard))
# Check the dataset structure using the str() function
str(model_data)
#We can also view the data frame via the View() function.
View(model_data)


# Spliting the dataset into training and test sets
# Using 70% for training and 30% for test
n <- round(nrow(model_data) * .70)
# Use the variable n to determine the size of our training sample
training_sample <- sample(seq_len(nrow(model_data)), size = n)
# Generate training and test sets
training <- model_data[training_sample, ]
test <- model_data[-training_sample, ]

# Train the classification tree model
churn_model <- rpart(formula = Exited ~.,data = training, method = "class")
View(churn_model)

# Using new model (churn_model) to predict on the test data
prediction <- predict(object = churn_model, newdata = test, type = "class")
hist(as.numeric(prediction)-1)

# Generating a confusion matrix for the prediction on the test data
confusionMatrix(data = prediction,reference = test$Exited,positive = '1')

# K-fold cross-validation 
set.seed(200)
cvdata <- trainControl(method = "cv", number = 10, savePredictions=TRUE)
model_fit <- train(Exited ~., data = model_data, method = 'rpart2', trControl=cvdata, tuneLength = 15)
model_fit

pred <- model_fit$pred
pred$equal <- ifelse(pred$pred == pred$obs, 1,0)
eachfold <- pred %>%  group_by(Resample) %>%summarise_at(vars(equal),list(Accuracy = mean))
eachfold

# Single-Predictor PDPs 
p1 <- partial(churn_model, pred.var = "Age", plot = TRUE,plot.engine = "ggplot2")
grid.arrange(p1, ncol =1)


# Multi PDP
# Computing partial dependence data for lstat and rm
pd <- partial(churn_model, pred.var = c("Age", "EstimatedSalary"))

# Adding contour lines and using a different color palette
rwb <- colorRampPalette(c("red", "white", "blue"))
pdp2 <- plotPartial(pd, contour = TRUE, col.regions = rwb)

# 3-D surface
pdp3 <- plotPartial(pd, levelplot = FALSE, zlab = "Exited", colorkey = TRUE,  screen = list(z = -20, x =
                                                                                              -60))

grid.arrange(pdp2,pdp3, ncol = 2)




