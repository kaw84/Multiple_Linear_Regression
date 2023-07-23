import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def make_scatter_plots_against_outcome(dataframe, columns_to_plot, outcome_column):
    """Make scatter plots for specified columns against outcome column."""
    for column in columns_to_plot:
        fig, axis = plt.subplots()
        axis.scatter(dataframe[column], dataframe[outcome_column])
        axis.set_ylabel(outcome_column)
        axis.set_xlabel(column)
        fig.suptitle(f"{column} vs {outcome_column}")
        plt.show()


def split_data_into_sets(features, outcome, train_size):
    """Split data into training and testing sets."""
    (
        features_train, features_test, outcome_train, outcome_test
    ) = train_test_split(features, outcome, train_size=train_size)

    return features_train, features_test, outcome_train, outcome_test


def create_model(features_train, outcome_train):
    """Create and train model."""

    model = LinearRegression()
    model.fit(features_train, outcome_train)

    return model


def compare_model_prediction_with_test_data(model, features_test, outcome_test, outcome_name):
    """Compare model prediction with test data."""
    print(model.score(features_test, outcome_test))

    prediction = model.predict(features_test)
    plt.scatter(outcome_test, prediction, alpha = 0.4)
    plt.ylabel(outcome_name)
    plt.xlabel("Predictions")
    plt.suptitle(f"Predictions vs {outcome_name}")
    plt.show()


# load and investigate the data here
player_stats = pd.read_csv('tennis_stats.csv')

# choosing input columns
columns_to_plot = [
    'FirstServe',
    'FirstServePointsWon',
    'FirstServeReturnPointsWon',
    'SecondServePointsWon',
    'SecondServeReturnPointsWon',
    'Aces',
    'BreakPointsConverted',
    'BreakPointsFaced',
    'BreakPointsOpportunities',
    'BreakPointsSaved',
    'DoubleFaults',
    'ReturnGamesPlayed',
    'ReturnGamesWon',
    'ReturnPointsWon',
    'ServiceGamesPlayed',
    'ServiceGamesWon',
    'TotalPointsWon',
    'TotalServicePointsWon',
]

# plotting multiple subplots to see relationship between input and Wins
make_scatter_plots_against_outcome(player_stats, columns_to_plot, 'Wins')

# selecting features contributing to wins based on the plots, linear
features_wins = player_stats[[
    'Aces',
    'BreakPointsFaced',
    'BreakPointsOpportunities',
    'DoubleFaults',
    'ReturnGamesPlayed',
    'ServiceGamesPlayed'
]]
outcome_wins = player_stats[['Wins']]

# splitting the data
(
    wins_features_train, wins_features_test, wins_outcome_train, wins_outcome_test
) = split_data_into_sets(features_wins, outcome_wins, 0.2)

# creating a model
wins_model = create_model(wins_features_train, wins_outcome_train)

#comparing a model with results
compare_model_prediction_with_test_data(
    wins_model, wins_features_test, wins_outcome_test, "Wins")

# ANALYSIS OF WINNINGS

# making a plot of features vs winnings
make_scatter_plots_against_outcome(player_stats, columns_to_plot, "Winnings")

# Choosing best features, linear and skewed
features_winnings = player_stats[[
    'Aces',
    'BreakPointsFaced',
    'FirstServePointsWon',
    'ReturnPointsWon',
    'ServiceGamesWon',
    'TotalServicePointsWon',
    'BreakPointsOpportunities',
    'DoubleFaults',
    'ReturnGamesPlayed',
    'ServiceGamesPlayed'
]]
outcome_winnings = player_stats[['Winnings']]

# splitting the data into training and test sets
(
    winnings_features_train, winnings_features_test, winnings_outcome_train, winnings_outcome_test
) = split_data_into_sets(features_winnings, outcome_winnings, 0.2)

# creating a model
winnings_model = create_model(winnings_features_train, winnings_outcome_train)

# comparing model with test set
compare_model_prediction_with_test_data(
    winnings_model, winnings_features_test, winnings_outcome_test, "Winnings")





