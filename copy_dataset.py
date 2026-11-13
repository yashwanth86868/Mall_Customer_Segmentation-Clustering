import shutil
import os

os.makedirs("data", exist_ok=True)

shutil.copy(
    r'C:\Users\ybali\Downloads\Mall_Customers.csv',
    "data/Mall_Customers.csv"
)

print("CSV file copied successfully!")