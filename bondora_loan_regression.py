import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from math import sqrt
from scipy import stats

# Load the dataset
df_raw = pd.read_csv("LoanData.csv")

# Define relevant columns
selected_columns = [
    "LoanId",
    "NewCreditCustomer",
    "LoanDate",
    "VerificationType",
    "Age",
    "Gender",
    "Country",
    "AppliedAmount",
    "Amount",
    "Interest",
    "LoanDuration",
    "Education",
    "EmploymentDurationCurrentEmployer",
    "HomeOwnershipType",
    "IncomeTotal",
    "ExistingLiabilities",
    "LiabilitiesTotal",
    "RefinanceLiabilities",
    "Rating",
    "NoOfPreviousLoansBeforeLoan",
    "AmountOfPreviousLoansBeforeLoan",
    "PreviousEarlyRepaymentsCountBeforeLoan"
]

# Select only the relevant columns
df_selected = df_raw[selected_columns].copy()

# Handle missing values
df_cleaned = df_selected.copy()

# Fill categorical columns with mode or specific values
df_cleaned["EmploymentDurationCurrentEmployer"].fillna("Unknown", inplace=True)
df_cleaned["HomeOwnershipType"].fillna("Other", inplace=True)
df_cleaned["Education"].fillna(df_cleaned["Education"].mode()[0], inplace=True)
df_cleaned["VerificationType"].fillna("Unknown", inplace=True)
df_cleaned["Gender"].fillna(df_cleaned["Gender"].mode()[0], inplace=True)

# Fill numerical columns with 0
num_cols_to_fill_zero = [
    "PreviousEarlyRepaymentsCountBeforeLoan",
    "AmountOfPreviousLoansBeforeLoan",
    "NoOfPreviousLoansBeforeLoan"
]
for col in num_cols_to_fill_zero:
    df_cleaned[col].fillna(0, inplace=True)

# Fill Rating with mode
df_cleaned["Rating"].fillna(df_cleaned["Rating"].mode()[0], inplace=True)

# Preview cleaned data
print(df_cleaned.head())

# Find the sample mean and standard deviation of the "Interest" column
interest_mean = df_cleaned["Interest"].mean()
interest_std = df_cleaned["Interest"].std()

# Find the number of borrowers that received a smaller "Amount" than they applied for
num_less_amount = (df_cleaned["Amount"] < df_cleaned["AppliedAmount"]).sum()

# Find the proportion of each loan rating
rating_proportion = df_cleaned["Rating"].value_counts() / len(df_cleaned)

# Print the results
print("Interest rate mean:", round(interest_mean, 2))
print("Interest rate standard deviation:", round(interest_std, 2))
print("Number of borrowers that received less money than they asked for:", num_less_amount)
print("Proportion of loan ratings:\n", rating_proportion.sort_index())

# Create the column "DebtToIncome"
df_cleaned["DebtToIncome"] = df_cleaned["Amount"] / df_cleaned["IncomeTotal"]

# Create a new column "IsRisky" that is True if the loan is risky
df_cleaned["IsRisky"] = (
    (df_cleaned["DebtToIncome"] >= 0.35) &
    df_cleaned["EmploymentDurationCurrentEmployer"].isin(["TrialPeriod", "UpTo1Year"])
)

# Calculate the proportion of risky loans
risky_proportion = df_cleaned["IsRisky"].sum() / len(df_cleaned)

# Calculate the mean interest rate of the risky loans
mean_interest_risky = df_cleaned[df_cleaned["IsRisky"]]["Interest"].mean()

# Calculate the mean interest rate of the non-risky loans
mean_interest_non_risky = df_cleaned[~df_cleaned["IsRisky"]]["Interest"].mean()

# Print the results
print("Risky loans proportion:", round(risky_proportion, 4))
print("Mean interest rate of risky loans:", round(mean_interest_risky, 4))
print("Mean interest rate of non-risky loans:", round(mean_interest_non_risky, 4))

scatterplot_fig = plt.figure(figsize=(12, 8))

# Create a list to store the correlation values
correlation = []

# List of segmentation columns
corr_columns = [
    "LoanDuration",
    "IncomeTotal",
    "AmountOfPreviousLoansBeforeLoan",
    "DebtToIncome"
]

# Iterate over the values 1-4 (one for each subplot)
for i in range(1, 5):
    # Get the column name
    column_name = corr_columns[i - 1]

    # Create subplot
    ax = scatterplot_fig.add_subplot(2, 2, i)

    # Create scatter plot
    ax.scatter(df_cleaned[column_name], df_cleaned["Interest"], alpha=0.6, color="royalblue")

    # Calculate and store the correlation
    corr = df_cleaned[column_name].corr(df_cleaned["Interest"])
    correlation.append(corr)

    # Set subplot title and labels
    ax.set_title(f"Interest vs {column_name}")
    ax.set_xlabel(column_name)
    ax.set_ylabel("Interest")

    # Print the result
    print(f"Correlation between Interest Rate and {column_name}:\n{round(corr, 4)}")

# General title
scatterplot_fig.suptitle(
    "Correlation between Interest Rate and Loan Duration, Total Income, \n"
    "Amount of Previous Loans, and Debt-to-Income",
    fontsize=14
)

plt.tight_layout()
plt.show()

# Get the number of samples where "AppliedAmount" differs from the approved "Amount"
num_differences = (df_cleaned["Amount"] < df_cleaned["AppliedAmount"]).sum()

# Get the total number of samples
n = len(df_cleaned)

# Calculate the sample proportion
phat = num_differences / n

# Calculate the standard error for proportions
se = sqrt(phat * (1 - phat) / n)

# Build the 95% confidence interval
confidence_interval = stats.norm.interval(0.95, loc=phat, scale=se)

# Print the results
print("The 95% confidence interval is", confidence_interval)

# Prepare predictors. Don't forget to add the constant term
X_simple = sm.add_constant(df_cleaned[["AmountOfPreviousLoansBeforeLoan"]])

# Select the dependent variable
Y_simple = df_cleaned["Interest"]

# Build the model
model_simple = sm.OLS(Y_simple, X_simple)

# Fit the model
results_simple = model_simple.fit()

# Print the results summary
print(results_simple.summary())

plt.figure()
sns.scatterplot(data=df_cleaned, x="AmountOfPreviousLoansBeforeLoan", y="Interest")

# Plot the regression line
plt.plot(
    df_cleaned["AmountOfPreviousLoansBeforeLoan"],
    results_simple.predict(X_simple),
    color="red"
)

# Labels and title
plt.title("Simple Linear Regression: Line of Best Fit")
plt.xlabel("Amount Of Previous Loans Before Loan")
plt.ylabel("Interest Rate")

plt.show()

# Define a list with the column names you want to use as predictors
predictors = [
    "AppliedAmount", "Amount",
    "IncomeTotal",
    "ExistingLiabilities", "RefinanceLiabilities",
    "Age", "NoOfPreviousLoansBeforeLoan", "AmountOfPreviousLoansBeforeLoan",
    "Education", "EmploymentDurationCurrentEmployer",
    "HomeOwnershipType", "Rating"
]

categorical = [
    "Education", "EmploymentDurationCurrentEmployer",
    "HomeOwnershipType", "Rating"
]

# Create the predictors dataframe with dummy variables and constant term
X = sm.add_constant(
    pd.get_dummies(
        df_cleaned[predictors],
        columns=categorical,
        drop_first=True,
        dtype=float
    )
)

# Create the target variable
Y = df_cleaned["Interest"]

# Create and fit the model
model = sm.OLS(Y, X)
results = model.fit()

print(results.summary())
