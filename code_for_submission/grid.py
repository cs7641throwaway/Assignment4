from mdptoolbox.example import grid_world
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import QLearning
import numpy as np
import random
import pandas as pd

#dims = [3, 10, 30]
dims = [100, 500, 1000, 10000]
ql_iter = {3: 3000000, 10: 50000000, 20: 200000000}

np.random.seed(0)
random.seed(0)

run_vi = True
run_pi = True
run_ql = False

# Seems to be really slow when using sparse implementation
# Stack overflow indicated it is due to more complicated mathematics with sparse
sparse = False

if run_vi and run_pi and run_ql:
	out_type = 'w'
else:
	out_type = 'a'

out_type = 'w'


def visualize_grid_world(R, X, Y):
	return(np.reshape(R, (Y, X)))


def visualize_policy(policy, dim):
	directions = {0: "v", 1: ">", 2: "^", 3: "<"}
	policy = [directions[policy_i] for policy_i in policy]
	policy_grid = np.reshape(policy, (dim, dim))
	return policy_grid


# Simulates to get the reward (returns avg and std_dev over num_repetitions)
def get_reward(P, R, policy, num_repetitions, dim, sparse):
	cum_reward_list = []
	if sparse:
		P_flat = np.ndarray((4, dim*dim, dim*dim))
		P_flat[0,] = P[0].todense()
		P_flat[1,] = P[1].todense()
		P_flat[2,] = P[2].todense()
		P_flat[3,] = P[3].todense()
		P = P_flat

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
	summary_file =  './output/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired)+'_'+str(prob_bad_state)+'_summary.rpt'
	pi_stats_file = './output/csv/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired)+'_'+str(prob_bad_state)+'_pi.csv'
	vi_stats_file = './output/csv/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired)+'_'+str(prob_bad_state)+'_vi.csv'
	ql_stats_file = './output/csv/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired)+'_'+str(prob_bad_state)+'_ql.csv'

	# Build world
	Trans_Prob, Rewards = grid_world(X=dim, Y=dim, prob_desired_move=prob_desired, prob_bad_state=prob_bad_state, is_sparse=sparse)
	gw = visualize_grid_world(Rewards[:,0], dim, dim)
	print("Grid world is: ")
	print(gw)
	with open(summary_file, out_type) as f:
		f.write("Grid world is:\n")
		f.write(str(gw)+"\n\n")

	if run_vi:
		vi = ValueIteration(Trans_Prob, Rewards, 0.9)
		vi_stats = vi.run()
		vi_df = pd.DataFrame(vi_stats)
		vi_df.to_csv(vi_stats_file, index_label="Iteration")
		reshaped_value_function = np.reshape(vi.V, (dim, dim))
		reshaped_policy = visualize_policy(vi.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, vi.policy, 10, dim, sparse)
		print("VI: Performed ", vi.iter, " iterations in ", vi.time, " and got rewards of: ", simulated_rewards)
		with open(summary_file, 'a') as f:
			f.write("***Value Iteration Section***\n")
			f.write("Iterations: "+str(vi.iter)+"\n")
			f.write("Runtime: "+str(vi.time)+"\n")
			f.write("Value function:\n")
			f.write(str(reshaped_value_function))
			f.write("\nPolicy:\n")
			f.write(str(reshaped_policy))
			f.write("\nSimulated rewards:\n")
			f.write(str(simulated_rewards))
			f.write("\n***End of Value Iteration Section***\n\n")

	if run_pi:
		pi = PolicyIteration(Trans_Prob, Rewards, 0.9)
		pi_stats = pi.run()
		pi_df = pd.DataFrame(pi_stats)
		pi_df.to_csv(pi_stats_file, index_label="Iteration")
		reshaped_value_function = np.reshape(pi.V, (dim, dim))
		reshaped_policy = visualize_policy(pi.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, pi.policy, 10, dim, sparse)
		print("PI: Performed ", pi.iter, " iterations in ", pi.time, " and got rewards of: ", simulated_rewards)
		with open(summary_file, 'a') as f:
			f.write("***Policy Iteration Section***\n")
			f.write("Iterations: "+str(pi.iter)+"\n")
			f.write("Runtime: "+str(pi.time)+"\n")
			f.write("Value function:\n")
			f.write(str(reshaped_value_function))
			f.write("\nPolicy:\n")
			f.write(str(reshaped_policy))
			f.write("\nSimulated rewards:\n")
			f.write(str(simulated_rewards))
			f.write("\n***End of Policy Iteration Section***\n\n")

	if run_ql:
		q_iter = ql_iter[dim]
		ql = QLearning(Trans_Prob, Rewards, 0.9, n_iter=q_iter)
		ql_stats = ql.run()
		ql_df = pd.DataFrame(ql_stats)
		ql_df.to_csv(ql_stats_file, index_label="Iteration")
		reshaped_value_function = np.reshape(ql.V, (dim, dim))
		reshaped_policy = visualize_policy(ql.policy, dim)
		simulated_rewards = get_reward(Trans_Prob, Rewards, ql.policy, 10, dim, sparse)
		print("QL: Performed ", ql.max_iter, " iterations in ", ql.time, " and got rewards of: ", simulated_rewards)
		with open(summary_file, 'a') as f:
			f.write("***QLearning Iteration Section***\n")
			f.write("Iterations: "+str(ql.max_iter)+"\n")
			f.write("Runtime: "+str(ql.time)+"\n")
			f.write("Value function:\n")
			f.write(str(reshaped_value_function))
			f.write("\nPolicy:\n")
			f.write(str(reshaped_policy))
			f.write("\nSimulated rewards:\n")
			f.write(str(simulated_rewards))
			f.write("\n***End of QLearning Section***\n\n")

