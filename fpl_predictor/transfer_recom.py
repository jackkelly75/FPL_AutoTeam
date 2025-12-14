#transfer reccomendations
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import itertools
import copy
import heapq
from typing import Dict, List, Tuple, Any

HORIZON_WEEKS = 7            # number of weeks to optimize over
BUDGET_CAP = 102.2
FREE_TRANSFERS_PER_WEEK = 1
MAX_BANKED_FREE = 5
PAID_TRANSFER_COST = 4       # points per extra transfer
BEAM_WIDTH = 30              # beam search width (increase for better results, slower)
MAX_PAID_TRANSFERS_TO_CONSIDER = 2  # per week when exploring (keeps branching manageable)
BENCH_WEIGHT = 0.15          # weight for bench predicted points in objective


def get_player_cost(df, player_name):
    """
    Get the cost (now_cost) of a player from the dataframe.
    """
    player_data = df[(df['player'] == player_name) & (df['gameweek'] == df['gameweek'].max())]
    if not player_data.empty:
        return player_data['now_cost'].iloc[0]  # FPL costs are in tenths
    return 0

def get_squad_positions(my_squad_15, df):
    """Get position information for each player in squad."""
    squad_positions = {}
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    for position, players in my_squad_15.items():
        for player in players:
            player_data = df[df['player'] == player]
            if not player_data.empty:
                position_id = player_data['element_type'].iloc[0]
                squad_positions[player] = position_map.get(position_id, position)
            else:
                squad_positions[player] = position
    
    return squad_positions

def get_initial_squad_cost(my_squad_15, df):
    """Calculate total cost of initial 15-player squad."""
    total_cost = 0
    for position, players in my_squad_15.items():
        for player in players:
            total_cost += get_player_cost(df, player)
    return total_cost

def get_available_replacements(current_squad_15, position, predictions_combined, df, 
                               max_cost_budget, squad_positions, all_players_in_predictions):
    """
    Get available players for transfer at a given position.
    Filter to players not already in squad and within budget.
    """
    current_position_players = current_squad_15[position] if position in current_squad_15 else []
    
    # Get all players of this position from predictions
    replacement_candidates = predictions_combined[
        (predictions_combined['position'] == position) &
        (~predictions_combined['player'].isin(all_players_in_predictions))
    ].drop_duplicates('player')
    
    # Calculate cost for each
    replacement_candidates['player_cost'] = replacement_candidates['player'].apply(
        lambda x: get_player_cost(df, x)
    )
    
    # Filter to within budget
    replacement_candidates = replacement_candidates[
        replacement_candidates['player_cost'] <= max_cost_budget
    ]
    
    return replacement_candidates.sort_values('player').drop_duplicates('player')

def calculate_squad_cost(squad_dict, df):
    """Calculate total cost of a squad structure."""
    total = 0
    for position, players in squad_dict.items():
        for player in players:
            total += get_player_cost(df, player)
    return total

def get_player_future_points(player_name, start_week, predictions_dict, remaining_weeks):
    """
    Get total predicted points for a player across remaining weeks.
    """
    total_points = 0
    for week in range(start_week, start_week + remaining_weeks):
        if week in predictions_dict:
            pred_df = predictions_dict[week]
            player_data = pred_df[pred_df['name'] == player_name]
            if not player_data.empty:
                total_points += player_data['mean'].values[0]
    return total_points


# --- HELPERS ---
def get_player_cost_map(player_cost_df: pd.DataFrame) -> Dict[str, float]:
    return dict(zip(player_cost_df['player_name'], player_cost_df['now_cost']))

def squad_total_cost(squad_15: Dict[str, List[str]], cost_map: Dict[str, float]) -> float:
    total = 0.0
    for pos_list in squad_15.values():
        for p in pos_list:
            total += cost_map.get(p, 0.0)
    return total

def flatten_squad(squad_15: Dict[str, List[str]]) -> List[str]:
    return [p for pos in ['GK','DEF','MID','FWD'] for p in squad_15[pos]]

def squad_valid_shape(squad_15: Dict[str, List[str]]) -> bool:
    return (len(squad_15['GK'])==2 and len(squad_15['DEF'])==5 and
            len(squad_15['MID'])==5 and len(squad_15['FWD'])==3)

# Precompute per-player per-week predictions and cumulative horizon sums
def build_prediction_maps(predictions_combined: pd.DataFrame, horizon_weeks: int) -> Tuple[dict, dict]:
    # weekly_map[(player,week)] = predicted_points_mean
    weekly_map = {}
    # cumulative_map[player] = sum predicted points over next horizon_weeks
    cumulative_map = {}
    players = predictions_combined['player'].unique()
    for p in players:
        player_df = predictions_combined[predictions_combined['player']==p]
        weekly_map.update({(p, int(r.week)): float(r['mean']) for _, r in player_df.iterrows()})
        # sum weeks 1..horizon_weeks if present
        cum = player_df[player_df['week'].between(1, horizon_weeks)]['mean'].sum()
        cumulative_map[p] = float(cum)
    return weekly_map, cumulative_map

# Evaluate a squad for a given week using your provided optimizer function
def evaluate_starting11_for_week(week: int, squad_15: Dict[str,List[str]], predictions_combined: pd.DataFrame,
                                 optimize_lineups_for_weeks, df=None, constraints=None) -> float:
    # filter predictions for that week
    week_df = predictions_combined[predictions_combined['week'] == week]
    optimized_lineups = optimize_lineups_for_weeks(week_df, squad_15, df, constraints)
    # as you said, total_points is at index [1]
    return float(optimized_lineups[1]['total_points'])

# Generate candidate replacements for each position based on cumulative predicted points
def candidate_replacements_by_position(squad_15: Dict[str,List[str]], cumulative_map: Dict[str,float],
                                       all_players: List[str], top_k=6) -> Dict[str, List[str]]:
    # For each position, return top_k players (not currently in squad) sorted by cumulative_map
    # We need a mapping from player -> position. We'll infer from squad lists and assume other players' positions
    # are derivable from predictions_combined or provided externally. For simplicity, require a mapping:
    raise NotImplementedError("You must provide a player->position mapping (player_position_map).")

# To keep the code self-contained, we'll require the user to supply player_position_map:
# player_position_map: dict player_name -> one of 'GK','DEF','MID','FWD'

# Generate plausible transfer actions for a given squad and week
def generate_transfer_actions(squad_15: Dict[str,List[str]], player_position_map: Dict[str,str],
                              cumulative_map: Dict[str,float], all_players: List[str],
                              cost_map: Dict[str,float], max_actions_per_pos=4) -> List[Tuple[Dict[str,List[str]], int]]:
    """
    Returns list of (new_squad_15, num_transfers_made) candidate actions.
    We generate:
      - no transfer (keep squad)
      - single transfers per position: replace one current squad player with a top candidate not in squad
      - optionally up to MAX_PAID_TRANSFERS_TO_CONSIDER simultaneous transfers (combinatorial)
    """
    candidates = []
    # no transfer
    candidates.append((copy.deepcopy(squad_15), 0))

    # Build list of available players by position sorted by cumulative_map
    available_by_pos = {'GK':[], 'DEF':[], 'MID':[], 'FWD':[]}
    for p in all_players:
        pos = player_position_map.get(p)
        if pos is None or pd.isna(pos) or str(pos).strip().upper() in {'', 'NA', 'N/A'}:
            continue
        available_by_pos[pos].append((p, cumulative_map.get(p, 0.0)))
    for pos in available_by_pos:
        available_by_pos[pos].sort(key=lambda x: x[1], reverse=True)

    # For each position, consider replacing each current squad player who is low value with top available candidates
    for pos in ['GK','DEF','MID','FWD']:
        current_players = squad_15[pos]
        # sort current players by cumulative predicted points ascending (worst first)
        current_sorted = sorted(current_players, key=lambda x: cumulative_map.get(x, 0.0))
        # top candidates not in squad
        top_candidates = [p for p,_ in available_by_pos[pos] if p not in flatten_squad(squad_15)]
        # limit candidates
        top_candidates = top_candidates[:max_actions_per_pos]
        # consider replacing up to 2 worst players in that position (to keep branching small)
        for out_player in current_sorted[:2]:
            for in_player in top_candidates:
                new_squad = copy.deepcopy(squad_15)
                # replace out_player with in_player
                new_squad[pos] = [in_player if x==out_player else x for x in new_squad[pos]]
                candidates.append((new_squad, 1))
    # Optionally consider pairwise combinations across positions (two transfers)
    # Keep it small: combine top single-transfer candidates
    single_transfers = [c for c in candidates if c[1]==1]
    for a,b in itertools.combinations(single_transfers, 2):
        # ensure they don't replace the same player or introduce duplicates
        s1, _ = a
        s2, _ = b
        # merge by applying both transfers to original squad
        merged = copy.deepcopy(squad_15)
        # find differences from original
        for pos in ['GK','DEF','MID','FWD']:
            # if s1 changed pos, adopt that change
            if set(s1[pos]) != set(squad_15[pos]):
                merged[pos] = s1[pos]
            if set(s2[pos]) != set(squad_15[pos]):
                merged[pos] = s2[pos]
        # validate no duplicates
        flat = flatten_squad(merged)
        if len(flat) == len(set(flat)) and squad_valid_shape(merged):
            candidates.append((merged, 2))
    # deduplicate by canonical tuple
    unique = {}
    for squad, ntrans in candidates:
        key = tuple(sorted(flatten_squad(squad)))
        if key not in unique or unique[key][1] > ntrans:
            unique[key] = (squad, ntrans)
    return list(unique.values())

# Beam-search planner
def plan_transfers_beam_search(my_squad_15: Dict[str,List[str]],
                               predictions_combined: pd.DataFrame,
                               player_cost: pd.DataFrame,
                               player_position_map: Dict[str,str],
                               optimize_lineups_for_weeks,
                               budget_cap = float, 
                               df=None, constraints=None,
                               afcon_bonus_week = None,
                               horizon_weeks: int = HORIZON_WEEKS,
                               beam_width: int = BEAM_WIDTH) -> Dict[str,Any]:
    # Precompute maps
    cost_map = get_player_cost_map(player_cost)
    weekly_map, cumulative_map = build_prediction_maps(predictions_combined, horizon_weeks)
    all_players = list(predictions_combined['player'].unique())
    print('generate all players')
    # State representation for beam: (score_so_far, week_index, squad_15, banked_free_transfers, transfers_used_total, history)
    # history: list of dicts per week with keys: 'week','action' (list of (out,in)), 'squad', 'week_starting11_points', 'bench_points', 'paid_transfers_used'
    initial_state = {
        'score': 0.0,
        'week': 1,
        'squad': copy.deepcopy(my_squad_15),
        'banked_free': 0,
        'transfers_used_total': 0,
        'history': []
    }

    beam = [initial_state]
    print("now predicting for each week")

    for week in range(1, horizon_weeks+1):
        print("week")
        next_beam = []
        for state in beam:
            print(state)
            squad = state['squad']
            banked_free = min(MAX_BANKED_FREE, state['banked_free'] + FREE_TRANSFERS_PER_WEEK)  # free transfers accrue at start of week
            if week == afcon_bonus_week:
                banked_free = 2 #5
            # generate candidate transfer actions
            actions = generate_transfer_actions(squad, player_position_map, cumulative_map, all_players, cost_map, max_actions_per_pos=6)
            print(actions)
            for new_squad, transfers_made in actions:
                # compute how many of those transfers are free vs paid
                free_used = min(banked_free, transfers_made)
                paid_used = max(0, transfers_made - free_used)
                new_banked_free = banked_free - free_used
                # check budget constraint
                total_cost = squad_total_cost(new_squad, cost_map)
                if total_cost > budget_cap:
                    # skip invalid squads
                    continue
                # evaluate this week's starting 11 points for new_squad
                try:
                    week_points = evaluate_starting11_for_week(week, new_squad, predictions_combined, optimize_lineups_for_weeks, df, constraints)
                except Exception as e:
                    # if optimizer fails, skip candidate
                    continue
                # compute bench points (approx): sum predicted points of bench players for this week * BENCH_WEIGHT
                # We need to identify bench players: starting 11 selection is done by optimizer; but we don't have the exact bench list.
                # As a proxy, compute sum of predicted points for all 15 minus starting11 points, then weight it.
                # First compute total predicted points for all 15 for this week:
                total_15_week = 0.0
                for p in flatten_squad(new_squad):
                    total_15_week += float(predictions_combined[(predictions_combined['player']==p) & (predictions_combined['week']==week)]['mean'].sum())
                bench_points_proxy = max(0.0, total_15_week - week_points) * BENCH_WEIGHT

                # penalty for paid transfers
                transfer_penalty = paid_used * PAID_TRANSFER_COST

                # new cumulative score
                new_score = state['score'] + week_points + bench_points_proxy - transfer_penalty

                # build new state
                new_state = {
                    'score': new_score,
                    'week': week+1,
                    'squad': copy.deepcopy(new_squad),
                    'banked_free': new_banked_free,
                    'transfers_used_total': state['transfers_used_total'] + transfers_made,
                    'history': state['history'] + [{
                        'week': week,
                        'action_transfers': transfers_made,
                        'paid_transfers_used': paid_used,
                        'free_transfers_used': free_used,
                        'squad': copy.deepcopy(new_squad),
                        'week_starting11_points': week_points,
                        'bench_points_proxy': bench_points_proxy,
                        'transfer_penalty': transfer_penalty,
                        'total_cost': total_cost
                    }]
                }
                next_beam.append(new_state)
        # keep top beam_width states by score
        if not next_beam:
            break
        next_beam.sort(key=lambda s: s['score'], reverse=True)
        beam = next_beam[:beam_width]

    # best final state
    best = max(beam, key=lambda s: s['score'])
    # post-process: for each week in history, compute the actual optimized starting 11 lineup (call optimizer again to get lineup details)
    detailed_history = []
    for h in best['history']:
        week = h['week']
        squad = h['squad']
        week_df = predictions_combined[predictions_combined['week']==week]
        optimized_lineups = optimize_lineups_for_weeks(week_df, squad, df, constraints)
        # attach lineup and total_points
        h_copy = copy.deepcopy(h)
        h_copy['optimized_lineup'] = optimized_lineups[1]  # as you described
        detailed_history.append(h_copy)

    return {
        'best_score': best['score'],
        'final_squad': best['squad'],
        'history': detailed_history
    }

