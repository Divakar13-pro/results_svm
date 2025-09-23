# Import libraries
import pandas as pd

# Load Titanic dataset from GitHub
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display first 5 rows
print(df.head())

# Optional: check dataset info
print(df.info())
