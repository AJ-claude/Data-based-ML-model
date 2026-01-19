# %%
pip install ydata-profiling

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# %%
from google.colab import files
uploaded = files.upload ()

# %%
churn_rate_analysis = pd.read_csv('Churn_Modelling_Dataset.csv')

# %%
churn_rate_analysis.info()
churn_rate_analysis.isnull().sum()
churn_rate_analysis.duplicated().sum()


# %%
churn_rate_analysis.profile_report()

# %%
churn_rate_analysis['Churn'].mean() * 100
#shows the percentage of customers who churn

# %%
churn_counts

# %%
# Calculate the counts of churned and non-churned customers
churn_counts = churn_rate_analysis['Churn'].value_counts()

# Create the bar chart
plt.figure(figsize=(5, 3))
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette='viridis', hue=churn_counts.index, legend=False)
plt.title('Distribution of Churn (0=No Churn, 1=Churn)')
plt.xlabel('Churn Status')
plt.ylabel('Number of Customers')
plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn'])
plt.show()


# Save the chart
plt.savefig('churn_distribution_bar_chart.png')
plt.show()

# %%
#checking for correlection across all parameters with respect to the churn value
churn_corr = [
    'Churn','CreditScore','Age','Tenure','Balance',
    'NumOfProducts','IsActiveMember','EstimatedSalary'
]
corr_analysis = churn_rate_analysis[churn_corr].corr()

churn_corr_analysis = corr_analysis['Churn'].drop('Churn').abs().sort_values(ascending=False)

print(churn_corr)


corr_analysis['Churn'].drop('Churn').loc[churn_corr_analysis.index]

# %%
churn_corr_analysis.plot(
    kind='bar',
    figsize=(8,5),
    title='Feature Correlation with Churn'
)

plt.ylabel('Absolute Correlation')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Save the chart
plt.savefig('Feature_Correlation_with_Churn.png')

plt.show()

# %% [markdown]
# This shows the top 3 factors are Age, Not active members and Balance

# %%
#Building a model that determines if a customer will churn on registration
features = [ "Age", "IsActiveMember","Balance", "NumOfProducts", "CreditScore", "Tenure", "EstimatedSalary"]
target = "Churn"
#x takes the parameters for a customer to churn and y is the predictive parameter the model uses
X = churn_rate_analysis[features]
y = churn_rate_analysis['Churn']

# %%
from sklearn.model_selection import train_test_split
#ensures the training and test cycles works well

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8
)

# %%
#this trains the model and take all values
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=700)
model.fit(X_train, y_train)


# %%
#this help the model predict the chrun rate of every new customers
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %%
#possible_churn_rate represent learnt pattern from previous factors common to
#churn_rate_analysis
possible_churn_rate = pd.Series(
    model.coef_[0],
    index=features
).sort_values()

print(possible_churn_rate)

# %%
#visual representation of the prediction
possible_churn_rate.plot(
    kind='barh',
    figsize=(8,5),
    title='Factors Influencing Customer Churn'
)
plt.xlabel('Impact on Churn')
plt.tight_layout()
plt.show()

# %%
coefficients = pd.Series(
    model.coef_[0],
    index=features
).sort_values(key=abs, ascending=False)

print(coefficients)



