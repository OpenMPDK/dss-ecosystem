import pandas as pd
import sys


file_path = str(sys.argv[1])
print(file_path)




df = pd.read_csv(file_path)

#print(df)



print("avg dataloading_time", df["dataloading_time"].mean())
print("avg training_time", df["training_time"].mean())
print("avg batch_time", df["batch_time"].mean())
print("sum dataloading_time", df["dataloading_time"].sum())
print("sum training_time", df["training_time"].sum())
print("sum batch_time", df["batch_time"].sum())


