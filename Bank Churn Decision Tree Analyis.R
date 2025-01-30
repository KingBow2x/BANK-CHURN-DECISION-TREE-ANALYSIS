# BANK CHURN DECISION TREE ANALYSIS ############################################

# Install required packages
install.packages("rpart.plot")
install.packages("Hmisc")
install.packages("ggpubr")
install.packages("caret", dependencies = TRUE)
install.packages("e1071", dependencies = TRUE)
install.packages("pdp")
install.packages("ggplot2")
install.packages("gridExtra")

# Load libraries
library(rpart)
library(rpart.plot)
library(caret)
library(Hmisc)
library(ggpubr)
library(pdp)
library(ggplot2)
library(gridExtra)

# Load dataset
full_data <- read.csv("F:/HW University/Big Data Course/data/Bank customers churn.csv")

# Check dataset structure
str(full_data)

# Install and load additional package for data description
install.packages("psych")
library(psych)

# Summarise dataset
describe(full_data)

# Convert response variable to a factor
full_data$Exited <- as.factor(full_data$Exited)

# Plot numerical variables
n1 <- ggplot(full_data, aes(CreditScore, colour = Exited)) + geom_freqpoly(binwidth = 5, size = 1)
n2 <- ggplot(full_data, aes(Age, colour = Exited)) + geom_freqpoly(binwidth = 5, size = 1)
n3 <- ggplot(full_data, aes(Balance, colour = Exited)) + geom_freqpoly(binwidth = 10000, size = 1)
n4 <- ggplot(full_data, aes(EstimatedSalary, colour = Exited)) + geom_freqpoly(binwidth = 10000, size = 1)

# Arrange numerical plots in a grid
ggarrange(n1, n2, n3, n4, ncol = 2, nrow = 2)

# Plot categorical variables
c1 <- ggplot(full_data, aes(Gender, fill = Exited)) + geom_bar(position = "fill", stat = "count")
c2 <- ggplot(full_data, aes(Geography, fill = Exited)) + geom_bar(position = "fill", stat = "count")
c3 <- ggplot(full_data, aes(Tenure, fill = Exited)) + geom_bar(position = "fill", stat = "count")
c4 <- ggplot(full_data, aes(NumOfProducts, fill = Exited)) + geom_bar(position = "fill", stat = "count")
c5 <- ggplot(full_data, aes(IsActiveMember, fill = Exited)) + geom_bar(position = "fill", stat = "count")
c6 <- ggplot(full_data, aes(HasCrCard, fill = Exited)) + geom_bar(position = "fill", stat = "count")

# Arrange categorical plots in grids
ggarrange(c1, c2, c3, c4, ncol = 2, nrow = 2)
ggarrange(c5, c6, ncol = 2, nrow = 1)

# Remove unnecessary columns
model_data <- subset(full_data, select = -c(CustomerId, Surname, RowNumber, HasCrCard))

# Check structure of cleaned dataset
str(model_data)

# View dataset
View(model_data)

# Split dataset: 70% training, 30% test
n <- round(nrow(model_data) * 0.70)
training_sample <- sample(seq_len(nrow(model_data)), size = n)

training <- model_data[training_sample, ]
test <- model_data[-training_sample, ]

# Train decision tree model
churn_model <- rpart(Exited ~ ., data = training, method = "class")

# View model details
View(churn_model)

# Plot decision tree
rpart.plot(churn_model, type = 2, extra = 2, under = TRUE, fallen.leaves = FALSE)

# Make predictions on test data
prediction <- predict(churn_model, newdata = test, type = "class")

# Plot prediction distribution
hist(as.numeric(prediction) - 1)

# Evaluate model performance with confusion matrix
confusionMatrix(data = prediction, reference = test$Exited, positive = "1")

# Perform 10-fold cross-validation
set.seed(200)
cvdata <- trainControl(method = "cv", number = 10, savePredictions = TRUE)
model_fit <- train(Exited ~ ., data = model_data, method = "rpart2", trControl = cvdata, tuneLength = 15)

# View cross-validation results
model_fit

# Calculate accuracy per fold
pred <- model_fit$pred
pred$equal <- ifelse(pred$pred == pred$obs, 1, 0)
eachfold <- pred %>% group_by(Resample) %>% summarise_at(vars(equal), list(Accuracy = mean))
eachfold

# Partial dependence plots (PDP)
p1 <- partial(churn_model, pred.var = "Age", plot = TRUE, plot.engine = "ggplot2")

# Multi-variable PDP
pd <- partial(churn_model, pred.var = c("Age", "EstimatedSalary"))

# Plot PDP with contour lines and colour gradient
rwb <- colorRampPalette(c("red", "white", "blue"))
pdp2 <- plotPartial(pd, contour = TRUE, col.regions = rwb)

# 3D surface plot
pdp3 <- plotPartial(pd, levelplot = FALSE, zlab = "Exited", colorkey = TRUE, screen = list(z = -20, x = -60))

# Arrange PDP plots
grid.arrange(pdp2, pdp3, ncol = 2)
