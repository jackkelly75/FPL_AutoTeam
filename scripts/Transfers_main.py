# Functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import heapq
from typing import Dict, List, Tuple, Any
from decimal import Decimal, ROUND_DOWN
import sys
sys.path.append(r"C:\Users\jackk\Desktop\FPL")
from transfer_recom import *
from team_stats import *
from fixture_transfer import *
from import_data_func import *

# Import data
df = import_player_data()


#drop this bloke out
df = df[~((df.first_name == 'Callum') & (df.second_name == 'Wilson'))]




git clone https://github.com/yourname/fpl-predictor.git
cd fpl-predictor
pip install -e .




from fpl_predictor import run_predictions

results = run_predictions(gameweek=20)
print(results)





#Squad info
entry_id = 4311890   # replace with your entry id
left_in_budget = 1.1
gw = 16

my_squad_15 = {
        'GK': ['Petrović', 'Sánchez'],
        'DEF': ['J.Timber', 'Chalobah', 'Muñoz', 'Tarkowski', 'Van den Berg'],
        'MID': ['Saka', 'Enzo', 'J.Palhinha', 'Wilson', 'Kudus'],
        'FWD': ['Woltemade', 'Thiago', 'Haaland'],
    }
my_squad_15_array = ['Petrović', 'Sánchez', 'J.Timber', 'Chalobah', 'Muñoz', 'Tarkowski', 'Van den Berg',
                     'Saka', 'Enzo', 'J.Palhinha', 'Wilson', 'Kudus','Woltemade', 'Thiago', 'Haaland']

player_position_map = {
   'Petrović':'GK', 'Sánchez':'GK', 
   'J.Timber':'DEF', 'Chalobah':'DEF', 'Muñoz':'DEF', 'Tarkowski':'DEF', 'Van den Berg':'DEF', 
   'Saka':'MID',   'Enzo':'MID',   'J.Palhinha':'MID',   'Wilson':'MID',   'Kudus':'MID',
   'Woltemade':'FWD',     'Thiago':'FWD',     'Haaland':'FWD',  
 }

# ----- Cost of players
#cost of all players to buy now
player_cost = df[(df.gameweek == max(df.gameweek))][['player_name', 'now_cost']]
#cost of my players if sold
transfers = get_squad_worth(entry_id, df, my_squad_15_array)
squad_cost = round(transfers['sell_price'].sum(), 1)
squad_cost = squad_cost + left_in_budget


# Set constraints (customize as needed)
constraints = {
    'GK': 1,
    'DEF': {'min': 3, 'max': 5},
    'MID': {'min': 2, 'max': 5},
    'FWD': {'min': 1, 'max': 3}
}

# Example: get fixtures for gameweek 1
fixture_1 = fixtures_for_event(gw)
fixture_2 = fixtures_for_event(gw+1)
fixture_3 = fixtures_for_event(gw+2)
fixture_4 = fixtures_for_event(gw+3)
fixture_5 = fixtures_for_event(gw+4)
fixture_6 = fixtures_for_event(gw+5)
fixture_7 = fixtures_for_event(gw+6)

# ----------- Find predictiosn for next 7 weeks
print("=" * 80)
print("FANTASY PREMIER LEAGUE - GAMEWEEK 1 PREDICTIONS")
print("=" * 80)
#calculte team stats
#team_stats = calculate_team_stats(df)
team_stats, meta = calculate_team_stats_dixon_coles_ewma(df, use_rho=True, rho_init=0.0, halflife_gw=5.0)

'''
pred = meta['predict_fixture']('Man Utd','Arsenal')
print(pred['expected_goals_home'], pred['expected_goals_away'])
print(pred['probabilities']['home_win'], pred['probabilities']['draw'], pred['probabilities']['away_win'])
'''

# Run predictions for all fixtures
predictions_df_1 = predict_fixture_1(df, fixture_1,  team_stats, predict_fixture_func = meta['predict_fixture'], n_simulations=10000)
predictions_df_1[["ci_lower", "ci_upper"]] = predictions_df_1["ci_95"].apply(get_bounds)

predictions_df_2 = predict_fixture_1(df, fixture_2,  team_stats, predict_fixture_func = meta['predict_fixture'], n_simulations=10000)
predictions_df_2[["ci_lower", "ci_upper"]] = predictions_df_2["ci_95"].apply(get_bounds)

predictions_df_3 = predict_fixture_1(df, fixture_3,  team_stats, predict_fixture_func = meta['predict_fixture'], n_simulations=10000)
predictions_df_3[["ci_lower", "ci_upper"]] = predictions_df_3["ci_95"].apply(get_bounds)

predictions_df_4 = predict_fixture_1(df, fixture_4,  team_stats, predict_fixture_func = meta['predict_fixture'], n_simulations=10000)
predictions_df_4[["ci_lower", "ci_upper"]] = predictions_df_4["ci_95"].apply(get_bounds)

predictions_df_5 = predict_fixture_1(df, fixture_5,  team_stats, predict_fixture_func = meta['predict_fixture'], n_simulations=10000)
predictions_df_5[["ci_lower", "ci_upper"]] = predictions_df_5["ci_95"].apply(get_bounds)

predictions_df_6 = predict_fixture_1(df, fixture_6,  team_stats, predict_fixture_func = meta['predict_fixture'], n_simulations=10000)
predictions_df_6[["ci_lower", "ci_upper"]] = predictions_df_6["ci_95"].apply(get_bounds)

predictions_df_7 = predict_fixture_1(df, fixture_7,  team_stats, predict_fixture_func = meta['predict_fixture'], n_simulations=10000)
predictions_df_7[["ci_lower", "ci_upper"]] = predictions_df_7["ci_95"].apply(get_bounds)


#sort injuered or banned players for weeks out
#will ahve to decide if you think people will be out for certain number of weeks yourself if now known
#can also be used for cup of nations

predictions_df_1 = set_player_out(predictions_df_1, 'Longstaff')
predictions_df_2 = set_player_out(predictions_df_2, 'Longstaff')
predictions_df_3 = set_player_out(predictions_df_3, 'Longstaff')
predictions_df_4 = set_player_out(predictions_df_4, 'Longstaff')
predictions_df_5 = set_player_out(predictions_df_5, 'Longstaff')
predictions_df_6 = set_player_out(predictions_df_6, 'Longstaff')
predictions_df_7 = set_player_out(predictions_df_7, 'Longstaff')

predictions_df_1 = set_player_out(predictions_df_1, 'Muñoz')
predictions_df_2 = set_player_out(predictions_df_2, 'Muñoz')
predictions_df_3 = set_player_out(predictions_df_3, 'Muñoz')
predictions_df_4 = set_player_out(predictions_df_4, 'Muñoz')
predictions_df_5 = set_player_out(predictions_df_5, 'Muñoz')
predictions_df_6 = set_player_out(predictions_df_6, 'Muñoz')

#ACON
acon = ['Salah','Ouattara','Foster','Sarr','Ndiaye','Iwobi',
'Salah','Ait-Nouri','Marmoush','Mbeumo','Wissa','Aina',
'Wan-Bissaka','Diouf','Agbadou']

for player_acon in acon:
	predictions_df_2 = set_player_out(predictions_df_2, player_acon)
	predictions_df_3 = set_player_out(predictions_df_3, player_acon)
	predictions_df_4 = set_player_out(predictions_df_4, player_acon)



# Combine all predictions
predictions_dict = {
	1: predictions_df_1,
    2: predictions_df_2,
    3: predictions_df_3,
    4: predictions_df_4,
    5: predictions_df_5,
    6: predictions_df_6,
    7: predictions_df_7
}

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Overall stats
print(f"\nTotal players analyzed: {len(predictions_df_1)}")
print(f"Average predicted points per player: {predictions_df_1['mean'].mean():.2f}")
print(f"Expected points across all fixtures: {predictions_df_1['mean'].sum():.2f}")

# Best value picks (high points, high reliability)
print("\n" + "=" * 80)
print("TOP 5 PREDICTIONS (Highest Expected Points)")
print("=" * 80)

top_20 = predictions_df_1.nlargest(5, 'mean')
print(top_20[['player', 'position', 'team', 'opponent', 'venue', 'mean', 'ci_lower',
	'ci_upper', 'prob_double_digit']].to_string(index=False))


# Best upside plays (high ceiling)
print("\n" + "=" * 80)
print("HIGHEST UPSIDE (95th percentile ceiling)")
print("=" * 80)
upside = predictions_df_1.nlargest(3, 'ci_upper')
print(upside[['player', 'position', 'team', 'opponent', 'venue', 'mean', 'ci_lower',
	'ci_upper', 'prob_double_digit']].to_string(index=False))
'''
# By fixture
print("\n" + "=" * 80)
print("PREDICTIONS BY FIXTURE")
print("=" * 80)
for fixture in predictions_df_1['matchup'].unique():
    fixture_preds = predictions_df_1[predictions_df_1['matchup'] == fixture]
    print(f"\n{fixture}")
    print(f"  Expected total points: {fixture_preds['mean'].sum():.2f}")
    print(f"  Top 5 scorers:")
    top_5_fixture = fixture_preds.nlargest(5, 'mean')
    for _, player in top_5_fixture.iterrows():
        print(f"    {player['player']:30} ({player['position']:3}) - " +
              f"{player['mean']:5.1f} pts (CI: {player['ci_lower']:5.1f}-{player['ci_upper']:5.1f})")
# By team
print("\n" + "=" * 80)
print("EXPECTED POINTS BY TEAM")
print("=" * 80)
team_totals = predictions_df_1.groupby('team').agg({
    'mean': 'sum',
    'player': 'count'
}).rename(columns={'player': 'players'}).sort_values('mean', ascending=False)
print(team_totals.to_string())

# Export full predictions
print("\n" + "=" * 80)
print("FULL PREDICTIONS TABLE")
print("=" * 80)
output_cols = ['player_name', 'position', 'team', 'opponent', 'venue', 'predicted_points_mean', 
               'predicted_points_std', 'ci_lower', 'ci_upper', 'prob_zero_points', 'prob_positive_points', 
               'prob_double_digit', 'minutes_played_rate']
print(predictions_df_2[output_cols].sort_values('predicted_points_mean', ascending=False).to_string(index=False))
'''
# Optional: Save to CSV
# predictions_df_1.to_csv('fpl_gameweek_1_predictions.csv', index=False)
# print("\nPredictions saved to 'fpl_gameweek_1_predictions.csv'")


# Step 1: Calculate team offensive and defensive stats
print("Predicted results")
for fixture in fixture_1:
	pred = meta['predict_fixture'](fixture[0],fixture[1])
	print(f"#--- {fixture} ---#")
	print(f"{fixture[0]} win: {round(pred['probabilities']['home_win'], 2)}")
	print(f"Draw: {round(pred['probabilities']['draw'], 2)}")
	print(f"{fixture[1]} win: {round(pred['probabilities']['away_win'], 2)}")


# Optimise line up
optimized_lineups = optimize_lineups_for_weeks(predictions_dict, my_squad_15, df, constraints)





predictions_list = [predictions_df_1, predictions_df_2, predictions_df_3,
     predictions_df_4, predictions_df_5, predictions_df_6, predictions_df_7]

predictions_combined = []
for i, preds in enumerate(predictions_list, start=1):
    tmp = preds.copy()
    tmp['week'] = i
    predictions_combined.append(tmp)

predictions_combined = pd.concat(predictions_combined, ignore_index=True)




# --- CONFIGURABLE PARAMETERS ---
HORIZON_WEEKS = 7            # number of weeks to optimize over
BUDGET_CAP = squad_cost
FREE_TRANSFERS_PER_WEEK = 1
MAX_BANKED_FREE = 5
PAID_TRANSFER_COST = 4       # points per extra transfer
BEAM_WIDTH = 50             # beam search width (increase for better results, slower)
MAX_PAID_TRANSFERS_TO_CONSIDER = 5  # per week when exploring (keeps branching manageable)
BENCH_WEIGHT = 0.1          # weight for bench predicted points in objective

# --- USER-SUPPLIED OBJECTS (placeholders for your environment) ---
# my_squad_15: dict with keys 'GK','DEF','MID','FWD' -> lists of player names (15 total)
# predictions_combined: DataFrame with columns ['player_name','week','predicted_points_mean']
# player_cost: DataFrame with columns ['player_name','now_cost']
# optimize_lineups_for_weeks(predictions_df, squad_15, df, constraints) -> returns dict-like with [1]['total_points']

player_position_map = predictions_combined.set_index('player')['position'].to_dict()
plan = plan_transfers_beam_search(my_squad_15, predictions_combined, player_cost, player_position_map, optimize_lineups_for_weeks, afcon_bonus_week =1, horizon_weeks = HORIZON_WEEKS , beam_width = BEAM_WIDTH, budget_cap = BUDGET_CAP, df=df, constraints=constraints)

#edit 
# --- Notes and tuning ---
# - Increase BEAM_WIDTH for better search (slower).
# - Increase MAX_PAID_TRANSFERS_TO_CONSIDER if you want to explore more paid-transfer combos.
# - The candidate generation uses cumulative predicted points; you can refine it to use per-week matchups, fixture difficulty, or rotation risk.
# - If you have a reliable player->position mapping and a larger universe of players, consider filtering available candidates by team fixtures or minutes-played likelihood.
# - The bench proxy is a heuristic; if your optimize_lineups_for_weeks returns bench details, replace the bench proxy with exact bench points.


def compare_by_position(a, b):
    for role in sorted(set(a) | set(b)):
        list_a = a.get(role, [])
        list_b = b.get(role, [])
        # compare up to the longer list
        for i, (x, y) in enumerate(zip(list_a, list_b)):
            if x != y:
                print(f"{role}: {x} -> {y}")
        # handle extra items if lengths differ
        if len(list_a) > len(list_b):
            for extra in list_a[len(list_b):]:
                print(f"{role}: {extra} -> (missing)")
        elif len(list_b) > len(list_a):
            for extra in list_b[len(list_a):]:
                print(f"{role}: (missing) -> {extra}")

compare_by_position(my_squad_15, plan['history'][0]['squad'])

print("#-------- Starting 11--------#")
print(plan['history'][0]['optimized_lineup']['lineup'])
print("#-------- Captain--------#")
pd.set_option('display.max_columns', 4)
print(plan['history'][0]['optimized_lineup']['lineup'].sort_values('mean', ascending = False)[['player', 'mean', 'ci_95', 'prob_double_digit']].head(3))


compare_by_position(my_squad_15, plan['history'][0]['squad'])
print("----")
compare_by_position(plan['history'][0]['squad'], plan['history'][1]['squad'])
print("----")
compare_by_position(plan['history'][1]['squad'], plan['history'][2]['squad'])
print("----")
compare_by_position(plan['history'][2]['squad'], plan['history'][3]['squad'])
print("----")
compare_by_position(plan['history'][3]['squad'], plan['history'][4]['squad'])
print("----")
compare_by_position(plan['history'][4]['squad'], plan['history'][5]['squad'])
print("----")
compare_by_position(plan['history'][5]['squad'], plan['history'][6]['squad'])


my_squad_15 = {'GK': ['Petrović', 'Donnarumma'],
 'DEF': ['Senesi', 'Chalobah', 'Muñoz', 'Tarkowski', 'Van den Berg'],
 'MID': ['Saka', 'Enzo', 'J.Palhinha', 'Longstaff', 'Kudus'],
 'FWD': ['Woltemade', 'Thiago', 'Haaland']}

optimized_lineups = optimize_lineups_for_weeks(predictions_dict, my_squad_15, df, constraints)
