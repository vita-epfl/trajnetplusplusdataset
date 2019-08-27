from collections import defaultdict
import trajnettools
from trajnettools import SceneRow
import pandas as pd


data_file = 'data/raw/cff_dataset/cff_dataset/al_position2013-02-06.csv'
dest_file = 'data/raw/cff_dataset/cff_dataset/dest2013-02-06.csv'
# df = pd.read_csv(data_file, encoding = "ISO-8859-1")

df = pd.read_csv(data_file, encoding = "ISO-8859-1", delimiter=";", names=["time", "place", "x", "y", "p"])
destf = pd.DataFrame(columns=['ped', 'x', 'y'])

ped_list = df["p"].unique()

df_time_low = df.loc[df["time"] < "2013-02-06T17:00:00:000"]
df_time_high = df.loc[df["time"] >= "2013-02-06T17:00:00:000"]

i = 0
for ped in ped_list:
	# print(ped)
	# print("First Lap")
	first_lap = df_time_low.loc[df_time_low["p"] == ped].sort_values('time').tail(n=1)
	# print(first_lap.iloc[0]["x"])
	# print(first_lap.iloc[0]["y"])
	if not first_lap.empty:
		destf.loc[i] = [ped , first_lap.iloc[0]["x"], first_lap.iloc[0]["y"]]
		# print("Second Lap")
		i = i + 1
	second_lap = df_time_high.loc[df_time_high["p"] == ped].sort_values('time').tail(n=1)
	# print(second_lap.iloc[0]["x"])
	# print(second_lap.iloc[0]["y"])
	if not second_lap.empty:
		destf.loc[i] = [100000 + ped , second_lap.iloc[0]["x"], second_lap.iloc[0]["y"]]	
		# destf.append(pd.DataFrame({"ped":[100000 + ped], "x":[second_lap.iloc[0]["x"]], "y":[second_lap.iloc[0]["y"]]}))
		i = i + 1
		
destf.to_csv(dest_file)
# print(destf.head())

# df.loc[df["p"] == 1].last()

# df = df.sort_values(('p'))	

# for col in df.columns:
# 	print(col)

# print(df.tail())
# df_time = df.loc[df["time"] < "2013-02-06T17:00:00:000"]
# print(df_time.loc[df_time["p"] == 3].sort_values('time').head(n=5))

# df_sort1 = df_1.sort_values(('time'))
# print(df_sort1.tail(n=5))
# print(df.iloc[526108])
# print(df.loc[df["p"] == 12])