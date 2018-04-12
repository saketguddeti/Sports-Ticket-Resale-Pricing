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
    filename = path_to_json + games_json.iloc[i]
    loaded_dict = json.load(open(filename[0]))
    event_info = event_info.append(pd.DataFrame(loaded_dict))

for i in range(98, 194):
    filename = path_to_json + games_json.iloc[i]
    loaded_dict = json.load(open(filename[0]))
    market_event = market_event.append(pd.DataFrame(loaded_dict))

for i in range(194, 290):
    filename = path_to_json + games_json.iloc[i]
    loaded_dict = json.load(open(filename[0]))
    market_zone = market_zone.append(pd.DataFrame(loaded_dict))

## Lower sideline - Quantity 2
temp = market_zone['Zone_Name'].drop_duplicates()
lower_sideline = market_zone[market_zone['Zone_Name'] == "Lower Level Sideline Club"]
lower_sideline_2 = lower_sideline[lower_sideline['quantity'] == 2]

event_stats = lower_sideline_2.groupby('Event_ID').agg(['count'])

lower_sideline_2_event = lower_sideline_2[lower_sideline_2['Event_ID'] == 103137944]

lower_sideline_2_event = pd.merge(lower_sideline_2_event, event_info[["Event_ID","Event_Date"]], on = ["Event_ID"])

lower_sideline_2_event["Days_to_game"] = (pd.to_datetime(lower_sideline_2_event["Event_Date"]) - pd.to_datetime(lower_sideline_2_event["Capture_Date"])).dt.days

plot_data = lower_sideline_2_event[["Average_Ticket_Price","Event_ID","Max_Ticket_Price","Min_Ticket_Price","Total_Listings","Total_Tickets","Zone_Name","quantity","Days_to_game"]].drop_duplicates().sort_values("Days_to_game")


import matplotlib.pyplot as plt
plt.scatter(plot_data['Days_to_game'], plot_data['Average_Ticket_Price'])

## Plot of Average Price vs Days to Game
plt.plot(plot_data['Days_to_game'], plot_data['Average_Ticket_Price'])
plt.xlim(120,0)
plt.xlabel("Days to Game")
plt.ylabel("Average Ticket Price")

plt.plot(plot_data['Days_to_game'], plot_data['Min_Ticket_Price'])
plt.xlim(120,0)
plt.xlabel("Days to Game")
plt.ylabel("Minimum Ticket Price")


## Field Sideline - Quantity 2
field_sideline = market_zone[market_zone['Zone_Name'] == "Field Sideline"]
field_sideline_2 = field_sideline[field_sideline['quantity'] == 2]

event_stats_2 = field_sideline_2.groupby('Event_ID').agg(['count'])

field_sideline_2_event = field_sideline_2[field_sideline_2['Event_ID'] == 103442113]

field_sideline_2_event = pd.merge(field_sideline_2_event, event_info[["Event_ID","Event_Date"]], on = ["Event_ID"])

field_sideline_2_event["Days_to_game"] = (pd.to_datetime(field_sideline_2_event["Event_Date"]) - pd.to_datetime(field_sideline_2_event["Capture_Date"])).dt.days

plot_data_2 = field_sideline_2_event[["Average_Ticket_Price","Event_ID","Max_Ticket_Price","Min_Ticket_Price","Total_Listings","Total_Tickets","Zone_Name","quantity","Days_to_game"]].drop_duplicates().sort_values("Days_to_game")

## Plot of Average Price vs Days to Game
plt.plot(plot_data_2['Days_to_game'], plot_data_2['Average_Ticket_Price'])
plt.xlim(220, 160)
plt.xlabel("Days to Game")
plt.ylabel("Average Ticket Price")

plt.plot(plot_data_2['Days_to_game'], plot_data_2['Min_Ticket_Price'])
plt.xlim(220, 160)
plt.xlabel("Days to Game")
plt.ylabel("Minimum Ticket Price")


event_info["Home_Team"].drop_duplicates()


sideline = market_zone[market_zone['Zone_Name'] == "Lower Level Sideline Club"]
sideline_2 = sideline[sideline['quantity'] == 2]

event_stats_2 = sideline_2.groupby('Event_ID').agg(['count'])

sideline_2_event =sideline_2[sideline_2['Event_ID'] == 103137944]

sideline_2_event = pd.merge(sideline_2_event, event_info[["Event_ID","Event_Date"]], on = ["Event_ID"])

sideline_2_event["Days_to_game"] = (pd.to_datetime(sideline_2_event["Event_Date"]) - pd.to_datetime(sideline_2_event["Capture_Date"])).dt.days

plot_data_2 = sideline_2_event[["Average_Ticket_Price","Event_ID","Max_Ticket_Price","Min_Ticket_Price","Total_Listings","Total_Tickets","Zone_Name","quantity","Days_to_game"]].drop_duplicates().sort_values("Days_to_game")

## Plot of Average Price vs Days to Game
plt.plot(plot_data_2['Days_to_game'], plot_data_2['Total_Listings'])
plt.xlim(220,0)
plt.ylim(0, 150)
plt.xlabel("Days to Game")
plt.ylabel("listings")

plt.plot(plot_data_2['Days_to_game'], plot_data_2['Min_Ticket_Price'])
plt.xlim(220, 160)
plt.xlabel("Days to Game")
plt.ylabel("Minimum Ticket Price")
