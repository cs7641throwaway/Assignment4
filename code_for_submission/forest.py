from mdptoolbox.example import forest
from mdptoolbox.example import grid_world
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import QLearning
import numpy as np
import pandas as pd


# Controls what is run
run_vi = True
run_pi = True
run_ql = False

# MDP problem setup
prob_fire = 0.4
reward_wait = 4
reward_cut = 1
gamma = 0.5

dims = [20, 50, 100, 500, 1000, 5000, 10000]

# Output files
for num_states in dims:
	forest_pi_stats_file = 'output/csv/forest_'+str(num_states)+'_'+str(prob_fire)+'_gamma='+str(gamma)+'_stats_pi.csv'
	forest_vi_stats_file = 'output/csv/forest_'+str(num_states)+'_'+str(prob_fire)+'_gamma='+str(gamma)+'_stats_vi.csv'
	forest_ql_stats_file = 'output/csv/forest_'+str(num_states)+'_'+str(prob_fire)+'_gamma='+str(gamma)+'_stats_ql.csv'
	forest_summary_file = 'output/forest_'+str(num_states)+'_'+str(prob_fire)+'_gamma='+str(gamma)+'_summary.rpt'

	# Sets up consistent seed
	np.random.seed(0)

	# MDP
	Trans_Prob, Rewards = forest(num_states, reward_wait, reward_cut, prob_fire, is_sparse=True)

	# Value Iteration
	# Convergence is based off of the change in value function
	# V - V_prev, then do max value - min value (error) at which point if that is less than some threshold, then converged
	if run_vi:
		print("Running Value Iteration ...")
		vi = ValueIteration(Trans_Prob, Rewards, gamma)
		vi_stats = vi.run()
		vi_df = pd.DataFrame(vi_stats)
		vi_df.to_csv(forest_vi_stats_file, index_label="Iteration")
		with open(forest_summary_file, 'w') as f:
			f.write("***Value Iteration***\n")
			f.write("Num iters: "+str(vi.iter)+"\nRuntime: "+str(vi.time))
			f.write("Optimal value function:\n"+str(vi.V)+"\n")
			f.write("Optimal policy :\n"+str(vi.policy)+"\n")
			f.write("***End of Value Iteration***\n\n")
		print("Value Iteration complete.")
		print("Optimal value function: ", vi.V)
		print("Optimal policy: ", vi.policy)

	# Policy Iteration
	# Calculates the new policy and if it's the same as the old policy, we say it has converged
	# Probably OK to use error in value function here though
	if run_pi:
		print("Running Policy Iteration ...")
		pi = PolicyIteration(Trans_Prob, Rewards, gamma)
		pi_stats = pi.run()
		pi_df = pd.DataFrame(pi_stats)
		pi_df.to_csv(forest_pi_stats_file, index_label="Iteration")
		print("Policy Iteration complete.")
		print("Optimal policy: ", pi.policy)
		with open(forest_summary_file, 'a') as f:
			f.write("***Policy Iteration***\n")
			f.write("Num iters: "+str(pi.iter)+"\nRuntime: "+str(pi.time))
			f.write("Optimal value function:\n"+str(pi.V)+"\n")
			f.write("Optimal policy :\n"+str(pi.policy)+"\n")
			f.write("***End of Policy Iteration***\n\n")

	# QLearning
	# TODO: Epsilon, alpha, and decay experiments
	if run_ql:
		print("Running QLearning ...")
		ql = QLearning(Trans_Prob, Rewards, gamma, n_iter=50000000)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(forest_ql_stats_file, index_label="Iteration")
		print("QLearning complete.")
		print("Optimal policy: ", ql.policy)
		with open(forest_summary_file, 'a') as f:
			f.write("***QLearning***\n")
			f.write("Optimal value function:\n"+str(ql.V)+"\n")
			f.write("Optimal policy :\n"+str(ql.policy)+"\n")
			f.write("***End of QLearning***\n\n")
