import os, json
import pandas as pd

"""data = json.load(open("C:/Users/Sreekanth Varma/Downloads/TiqAssist/data/market_zone_level_2018-04-03.json"))
data = pd.DataFrame.from_dict(data)"""

path_to_json = 'C:/Users/Sreekanth Varma/Downloads/TiqAssist/data/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)  

games_json = pd.DataFrame.from_dict(json_files)

event_info = pd.DataFrame()
market_event = pd.DataFrame()
market_zone = pd.DataFrame()

for i in range(0, 98):
    filename = 'C:/Users/Sreekanth Varma/Downloads/TiqAssist/data/' + games_json.iloc[i]
    loaded_dict = json.load(open(filename[0]))
    event_info = event_info.append(pd.DataFrame(loaded_dict))

for i in range(98, 194):
    filename = 'C:/Users/Sreekanth Varma/Downloads/TiqAssist/data/' + games_json.iloc[i]
    loaded_dict = json.load(open(filename[0]))
    market_event = market_event.append(pd.DataFrame(loaded_dict))

for i in range(194, 290):
    filename = 'C:/Users/Sreekanth Varma/Downloads/TiqAssist/data/' + games_json.iloc[i]
    loaded_dict = json.load(open(filename[0]))
    market_zone = market_zone.append(pd.DataFrame(loaded_dict))


lower_sideline = market_zone[market_zone['Zone_Name'] == "Lower Level Sideline Club"]
lower_sideline_2 = lower_sideline[lower_sideline['quantity'] == 2]

