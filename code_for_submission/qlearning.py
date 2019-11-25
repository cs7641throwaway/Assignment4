from mdptoolbox.example import grid_world
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
ql_iter = 50000000

np.random.seed(0)
random.seed(0)

run_alpha_sweep = True
run_epsilon_sweep = True


def visualize_grid_world(R, X, Y):
	return(np.reshape(R, (Y, X)))


def visualize_policy(policy, dim):
	directions = {0: "v", 1: ">", 2: "^", 3: "<"}
	policy = [directions[policy_i] for policy_i in policy]
	policy_grid = np.reshape(policy, (dim, dim))
	return policy_grid


# Simulates to get the reward (returns avg and std_dev over num_repetitions)
def get_reward(P, R, policy, num_repetitions):
	cum_reward_list = []
	for i in range(num_repetitions):
		cum_rewards = 0
		#print("Simulation iteration: ", i)
		current_state = 0
		terminal_state = dim*dim-1
		while current_state != terminal_state:
			move = policy[current_state]
			transitions = P[move, current_state]
			cum_transitions = transitions.cumsum()
			rand = random.random()
			# TODO: Get first index where rand is less than cum_transitions; that is the next state
			next_state = np.argmax(cum_transitions>rand)
			reward = R[next_state, move]
			cum_rewards += reward
			current_state = next_state
		#print("Finished simulation: ", i, " with cumulative rewards of :", cum_rewards)
		cum_reward_list.append(cum_rewards)
	return cum_reward_list


for dim in dims:
	prob_desired = 0.9
	prob_bad_state = 0.1
	base_path = './output/csv/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired)+'_'+str(prob_bad_state)+'_'
	base_sweep_path = './output/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired)+'_'+str(prob_bad_state)+'_'

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


	# Build world
	Trans_Prob, Rewards = grid_world(X=dim, Y=dim, prob_desired_move=prob_desired, prob_bad_state=prob_bad_state)
	gw = visualize_grid_world(Rewards[:,0], dim, dim)
	print("Grid world is: ")
	print(gw)

	if run_alpha_sweep:
		with open(alpha_sweep_file, 'w') as f:
			f.write("Grid world is: \n")
			f.write(str(gw)+"\n\n")
		# Low
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, alpha=0.001, alpha_decay=1.00)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(alpha_low_file, index_label="Iteration")
		reshaped_policy = visualize_policy(ql.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, ql.policy, 10)
		with open(alpha_sweep_file, 'a') as f:
			f.write("***Alpha = 0.001 with No Decay***\n")
			f.write("Policy is:\n"+str(reshaped_policy)+"\n")
			f.write("Rewards are:\n"+str(simulated_rewards)+"\n")
			f.write("***End of Alpha = 0.001 with No Decay***\n\n")

		# High
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, alpha=0.5, alpha_decay=1.00)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(alpha_high_file, index_label="Iteration")
		reshaped_policy = visualize_policy(ql.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, ql.policy, 10)
		with open(alpha_sweep_file, 'a') as f:
			f.write("***Alpha = 0.5 with No Decay***\n")
			f.write("Policy is:\n"+str(reshaped_policy)+"\n")
			f.write("Rewards are:\n"+str(simulated_rewards)+"\n")
			f.write("***End of Alpha = 0.5 with No Decay***\n\n")

		# Decay
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, alpha=0.1)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(alpha_decay_file, index_label="Iteration")
		reshaped_policy = visualize_policy(ql.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, ql.policy, 10)
		with open(alpha_sweep_file, 'a') as f:
			f.write("***Alpha = 0.1 with Decay***\n")
			f.write("Policy is:\n"+str(reshaped_policy)+"\n")
			f.write("Rewards are:\n"+str(simulated_rewards)+"\n")
			f.write("***End of Alpha = 0.1 with Decay***\n\n")

	if run_epsilon_sweep:
		with open(epsilon_sweep_file, 'w') as f:
			f.write("Grid world is: \n")
			f.write(str(gw)+"\n\n")
		# Low
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, epsilon=0.1, epsilon_decay=1.00)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(epsilon_low_file, index_label="Iteration")
		reshaped_policy = visualize_policy(ql.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, ql.policy, 10)
		with open(epsilon_sweep_file, 'a') as f:
			f.write("***Epsilon = 0.1 with No Decay***\n")
			f.write("Policy is:\n"+str(reshaped_policy)+"\n")
			f.write("Rewards are:\n"+str(simulated_rewards)+"\n")
			f.write("***End of Epsilon = 0.001 with No Decay***\n\n")

		# High
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, epsilon=0.9, epsilon_decay=1.00)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(epsilon_high_file, index_label="Iteration")
		reshaped_policy = visualize_policy(ql.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, ql.policy, 10)
		with open(epsilon_sweep_file, 'a') as f:
			f.write("***Epsilon = 0.9 with No Decay***\n")
			f.write("Policy is:\n"+str(reshaped_policy)+"\n")
			f.write("Rewards are:\n"+str(simulated_rewards)+"\n")
			f.write("***End of Epsilon = 0.9 with No Decay***\n\n")

		# Decay
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=ql_iter, epsilon=1.0)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(epsilon_decay_file, index_label="Iteration")
		reshaped_policy = visualize_policy(ql.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, ql.policy, 10)
		with open(epsilon_sweep_file, 'a') as f:
			f.write("***Epsilon = 1.0 with Decay***\n")
			f.write("Policy is:\n"+str(reshaped_policy)+"\n")
			f.write("Rewards are:\n"+str(simulated_rewards)+"\n")
			f.write("***End of Epsilon = 0.1 with Decay***\n\n")

