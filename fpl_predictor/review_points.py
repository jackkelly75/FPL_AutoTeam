import requests, json
from pprint import pprint


base_url = 'https://fantasy.premierleague.com/api/entry/4311890/history/'
r = requests.get(base_url).json()
results_JK = pd.DataFrame(r['current'])
base_url = 'https://fantasy.premierleague.com/api/entry/4317623/history/'
r = requests.get(base_url).json()
results_SS = pd.DataFrame(r['current'])
base_url = 'https://fantasy.premierleague.com/api/entry/4838154/history/'
r = requests.get(base_url).json()
results_LH = pd.DataFrame(r['current'])
base_url = 'https://fantasy.premierleague.com/api/entry/2845955/history/'
r = requests.get(base_url).json()
results_SC = pd.DataFrame(r['current'])
base_url = 'https://fantasy.premierleague.com/api/entry/606597/history/'
r = requests.get(base_url).json()
results_MS = pd.DataFrame(r['current'])
base_url = 'https://fantasy.premierleague.com/api/entry/5658784/history/'
r = requests.get(base_url).json()
results_MR = pd.DataFrame(r['current'])



import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4.5))       # width, height in inches
plt.plot( results_JK['event'], results_JK['total_points'], color='tab:blue', linewidth=2, label='JK')
plt.plot( results_SS['event'], results_SS['total_points'], color='tab:red', linewidth=2, label='SS')
plt.plot( results_LH['event'], results_LH['total_points'], color='tab:purple', linewidth=2, label='LH')
plt.plot( results_SC['event'], results_SC['total_points'], linestyle=':', color='tab:blue', linewidth=2, label='SC')
plt.plot( results_MS['event'], results_MS['total_points'], linestyle=':', color='tab:red', linewidth=2, label='MS')
plt.plot( results_MR['event'], results_MR['total_points'], linestyle=':', color='tab:purple', linewidth=2, label='MR')
plt.title('Simple Line Plot')      # titl
plt.xlabel('X axis')               # x label
plt.ylabel('Y axis')               # y label
plt.grid(True, linestyle='--', alpha=0.6)  # grid for readability
plt.legend()                       # show legend
plt.tight_layout()
plt.show()                              


# --- Example usage ---
# You must provide player_position_map: dict mapping every player in predictions_combined to their position.
# Example:
# player_position_map = {
#   'Petrović':'GK', 'Donnarumma':'GK', 'Senesi':'DEF', ...,
#   'SomeOtherPlayer':'MID', ...
# }




# Then call:
# plan = plan_transfers_beam_search(my_squad_15, predictions_combined, player_cost, player_position_map, optimize_lineups_for_weeks, df=None, constraints=None)
# The returned 'plan' contains 'best_score', 'final_squad', and 'history' with week-by-week details.

# --- Notes and tuning ---
# - Increase BEAM_WIDTH for better search (slower).
# - Increase MAX_PAID_TRANSFERS_TO_CONSIDER if you want to explore more paid-transfer combos.
# - The candidate generation uses cumulative predicted points; you can refine it to use per-week matchups, fixture difficulty, or rotation risk.
# - If you have a reliable player->position mapping and a larger universe of players, consider filtering available candidates by team fixtures or minutes-played likelihood.
# - The bench proxy is a heuristic; if your optimize_lineups_for_weeks returns bench details, replace the bench proxy with exact bench points.