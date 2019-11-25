from mdptoolbox.example import forest
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import QLearning
import numpy as np
import random
import pandas as pd

# Goal of this is to see impact of sweeping learning rate & epsilon
# For alpha, probably good enough to just show convergence rates
# For epsilon, probably want to show both learning rates as well as most frequently visited states?
dims = [10]
ql_iter = 5000000

np.random.seed(0)
random.seed(0)

run_alpha_sweep = False
run_epsilon_sweep = False
run_gamma_sweep = True


# Simulates to get the reward (returns avg and std_dev over num_repetitions)

for dim in dims:
	prob_fire = 0.4
	base_path = './output/csv/forest_'+str(dim)+'_'+str(prob_fire)+'_'
	base_sweep_path = './output/forest_'+str(dim)+'_'+str(prob_fire)+'_'

	# Alpha files
	alpha_low_file = base_path+'alpha_0.001_no_decay.csv'
	alpha_high_file = base_path+'alpha_0.5_no_decay.csv'
	alpha_decay_file = base_path+'alpha_0.1_decay.csv'
	alpha_sweep_file = base_sweep_path+'alpha_sweep.rpt'

	# Epsilon files
	epsilon_low_file = base_path+'epsilon_0.1_no_decay.csv'
	epsilon_high_file = base_path+'epsilon_0.9_no_decay.csv'
	epsilon_decay_file = base_path+'epsilon_1.0_decay.csv'
	epsilon_sweep_file = base_sweep_path+'epsilon_sweep.rpt'

	# Gamma files
	gamma_low_file = base_path+'gamma_0.1.csv'
	gamma_med_file = base_path+'gamma_0.5.csv'
	gamma_high_file = base_path+'gamma_0.9.csv'
	gamma_sweep_file = base_sweep_path+'gamma_sweep.rpt'



	# Build world
	Trans_Prob, Rewards = forest(dim, 4, 1, 0.4, is_sparse=False)

	if run_alpha_sweep:
		# Low
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, alpha=0.001, alpha_decay=1.00)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(alpha_low_file, index_label="Iteration")
		with open(alpha_sweep_file, 'a') as f:
			f.write("***Alpha = 0.001 with No Decay***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Alpha = 0.001 with No Decay***\n\n")

		# High
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, alpha=0.5, alpha_decay=1.00)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(alpha_high_file, index_label="Iteration")
		with open(alpha_sweep_file, 'a') as f:
			f.write("***Alpha = 0.5 with No Decay***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Alpha = 0.5 with No Decay***\n\n")

		# Decay
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, alpha=0.1, alpha_decay=0.99999)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(alpha_decay_file, index_label="Iteration")
		with open(alpha_sweep_file, 'a') as f:
			f.write("***Alpha = 0.1 with Decay***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Alpha = 0.1 with Decay***\n\n")

	if run_epsilon_sweep:
		# Low
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, epsilon=0.1, epsilon_decay=1.00)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(epsilon_low_file, index_label="Iteration")
		with open(epsilon_sweep_file, 'a') as f:
			f.write("***Epsilon = 0.1 with No Decay***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Epsilon = 0.1 with No Decay***\n\n")

		# High
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, epsilon=0.9, epsilon_decay=1.00)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(epsilon_high_file, index_label="Iteration")
		with open(epsilon_sweep_file, 'a') as f:
			f.write("***Epsilon = 0.9 with No Decay***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Epsilon = 0.9 with No Decay***\n\n")

		# Decay
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, epsilon=1.0, epsilon_decay=0.99999)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(epsilon_decay_file, index_label="Iteration")
		with open(epsilon_sweep_file, 'a') as f:
			f.write("***Epsilon = 1.0 with Decay***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Epsilon = 1.0 with Decay***\n\n")

	if run_gamma_sweep:
		# Low
		ql = QLearning(Trans_Prob, Rewards, 0.1, n_iter=ql_iter)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(gamma_low_file, index_label="Iteration")
		with open(gamma_sweep_file, 'a') as f:
			f.write("***Gamma = 0.1***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Gamma = 0.1***\n\n")

		# Med
		ql = QLearning(Trans_Prob, Rewards, 0.5, n_iter=ql_iter)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(gamma_med_file, index_label="Iteration")
		with open(gamma_sweep_file, 'a') as f:
			f.write("***Gamma = 0.5***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Gamma = 0.5***\n\n")

		# High
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(gamma_high_file, index_label="Iteration")
		with open(gamma_sweep_file, 'a') as f:
			f.write("***Gamma = 0.9***\n")
			f.write("Policy is:\n"+str(ql.policy)+"\n")
			f.write("***End of Gamma = 0.9***\n\n")
