import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Import the data
nhl_df = pd.read_csv("NHL_Cleaned.csv")

# Check if the goalie is in the dictionary
def In_Dict(shot, dict):
    for name in dict:
        if nhl_df["goalieNameForShot"][shot] == name:
            return True

    return False

# Plot the shots of a goalie
# https://towardsdatascience.com/nhl-analytics-with-python-6390c5d3206d
def Goalie_Save_Plot(data, type = "all"):
    xbnds = np.array([0, 100])
    ybnds = np.array([-50, 50])
    extent = [xbnds[0], xbnds[1], ybnds[0], ybnds[1]]

    # Create the scatter over image plot
    fig = plt.figure(figsize = (10, 10))
    ax = plt.subplot(111)
    ax.set_facecolor("white")
    ax.patch.set_facecolor("white")
    ax.patch.set_alpha(0.0)
    ax.set_title("Saves in Blue and Goals in Orange", fontsize = 15)
    plt.suptitle("Shots by Arena Location", fontsize = 20)
    ax.set_xticklabels(labels = [""], fontsize = 18, alpha = 0.7, minor = False)
    ax.set_yticklabels(labels = [""], fontsize = 18, alpha = 0.7, minor = False)
    image = Image.open("NHL_Rink_Half.png")
    image = image.rotate(90)
    ax.imshow(image)
    width, height = image.size

    # Location lists
    x_loc_save = []
    y_loc_save = []
    x_loc_goal = []
    y_loc_goal = []

    # Get the shot locations for the player
    for loc in range(len(data)):
        if data[loc][2] == 0:
            x_loc_save.append(data[loc][0] * width / 100)
            y_loc_save.append((data[loc][1] + 50) * height / 100)
        else:
            x_loc_goal.append(data[loc][0] * width / 100)
            y_loc_goal.append((data[loc][1] + 50) * height / 100)

    # Filter by the shot given by the user
    if type == "all":
        plt.scatter(x_loc_save, y_loc_save, c = "blue")
        plt.scatter(x_loc_goal, y_loc_goal, c = "orange")
    elif type == "goals":
        plt.scatter(x_loc_goal, y_loc_goal, c = "orange")
    elif type == "saves":
        plt.scatter(x_loc_save, y_loc_save, c = "blue")

    plt.show()

# Create a bar graph by shot type
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
def Shot_Type(data, type = "all"):
    # Lists to hold amount of shots per shot type
    shot_counts_saves = [0 for i in range(7)]
    shot_counts_goals = [0 for i in range(7)]

    # Change to dataframe to make use of dataframe capabilities
    data_df = pd.DataFrame(data)

    # Sum the values for each shot for both saves and goals
    # https://stackoverflow.com/questions/28236305/how-do-i-sum-values-in-a-column-that-match-a-given-condition-using-pandas
    for col in range(3, len(data_df.columns)):
        shot_counts_saves[col - 3] = data_df.loc[data_df[2] == 0, col].sum()
        shot_counts_goals[col - 3] = data_df.loc[data_df[2] == 1, col].sum()

    fig, ax = plt.subplots()

    # Beautify the graph
    ax.set_ylabel('Shots')
    ax.set_title('Shots by Type')
    labels = ["Back Shot", "Deflection Shot", "Slap Shot", "Snap Shot", "Tip Shot", "Wrap Shot", "Wrist Shot"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation = 45)
    ax.legend()

    # Filter by the shot given by the user
    if type == "all":
        saves = ax.bar(np.arange(len(labels)) - 0.35 / 2, shot_counts_saves, 0.3, label = "Saves", color = "blue")
        goals = ax.bar(np.arange(len(labels)) + 0.35 / 2, shot_counts_goals, 0.3, label = "Goals", color = "orange")
    elif type == "goals":
        goals = ax.bar(np.arange(len(labels)) + 0.35 / 2, shot_counts_goals, 0.3, label = "Goals", color = "orange")
    elif type == "saves":
        saves = ax.bar(np.arange(len(labels)) + 0.35 / 2, shot_counts_saves, 0.3, label = "Saves", color = "blue")

    fig.tight_layout()

    plt.show()

def Percent_Of_Shots(data):
    # Lists to hold amount of shots per shot type
    shot_counts_saves = [0 for i in range(7)]
    shot_counts_goals = [0 for i in range(7)]
    shot_counts_total = [0 for i in range(7)]

    # Convert to dataframe to use dataframe capabilities
    data_df = pd.DataFrame(data)

    # Sum the values for each shot for both saves and goals
    # https://stackoverflow.com/questions/28236305/how-do-i-sum-values-in-a-column-that-match-a-given-condition-using-pandas
    for col in range(3, len(data_df.columns)):
        shot_counts_saves[col - 3] = data_df.loc[data_df[2] == 0, col].sum()
        shot_counts_goals[col - 3] = data_df.loc[data_df[2] == 1, col].sum()

    # Get the effectiveness of each shot type
    for col in range(len(shot_counts_saves)):
        shot_counts_total[col] = shot_counts_goals[col] / (shot_counts_saves[col] + shot_counts_goals[col]) * 100

    fig, ax = plt.subplots()

    labels = ["Back Shot", "Deflection Shot", "Slap Shot", "Snap Shot", "Tip Shot", "Wrap Shot", "Wrist Shot"]

    # Beautify the graph
    ax.set_ylabel('Shots')
    ax.set_title('Percentage Shots by Type')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation = 15)
    ax.bar(np.arange(len(labels)), shot_counts_total, 0.5, label = "Saves Out of Total", color = "orange")

    plt.show()

# Create an image map of shot locations
shots_on_goalie = {}

# Iterate over every shot in the dataframe
for shot in range(len(nhl_df)):
    exists = In_Dict(shot, shots_on_goalie)
    # Check if the goalie for the shot is in the dictionary
    if exists:
        shots_on_goalie[nhl_df["goalieNameForShot"][shot]].append((nhl_df["arenaAdjustedXCordABS"][shot], nhl_df["yCord"][shot], nhl_df["goal"][shot],
                                                                nhl_df["shotType__BACK"][shot], nhl_df["shotType__DEFL"][shot], nhl_df["shotType__SLAP"][shot],
                                                                nhl_df["shotType__SNAP"][shot], nhl_df["shotType__TIP"][shot], nhl_df["shotType__WRAP"][shot],
                                                                nhl_df["shotType__WRIST"][shot]))
    else:
        shots_on_goalie[nhl_df["goalieNameForShot"][shot]] = [(nhl_df["arenaAdjustedXCordABS"][shot], nhl_df["yCord"][shot], nhl_df["goal"][shot],
                                                                nhl_df["shotType__BACK"][shot], nhl_df["shotType__DEFL"][shot], nhl_df["shotType__SLAP"][shot],
                                                                nhl_df["shotType__SNAP"][shot], nhl_df["shotType__TIP"][shot], nhl_df["shotType__WRAP"][shot],
                                                                nhl_df["shotType__WRIST"][shot])]

# Setup the Streamlit app
st.title("Shot Statistics by Goalie")
st.sidebar.title("Filters")
goalie = st.sidebar.selectbox("Choose a Goalie", list(shots_on_goalie.keys()))
save = st.sidebar.selectbox("Choose a Shot Type", ["all", "saves", "goals"])
# Create the graphs
scatter_image = st.pyplot(Goalie_Save_Plot(shots_on_goalie[goalie], save))
bar_image = st.pyplot(Shot_Type(shots_on_goalie[goalie], save))
bar_of_total_image = st.pyplot(Percent_Of_Shots(shots_on_goalie[goalie]))
