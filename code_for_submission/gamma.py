from mdptoolbox.example import forest
from mdptoolbox.example import grid_world
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.mdp import ValueIteration
import numpy as np
import random
import pandas as pd

# Goal of this is to see impact of sweeping learning rate & epsilon
# For alpha, probably good enough to just show convergence rates
# For epsilon, probably want to show both learning rates as well as most frequently visited states?

#grid_dims = [10, 30, 50]
#forest_dims = [10, 500, 1000]
grid_dims = [10]
forest_dims = [10]

np.random.seed(0)
random.seed(0)

run_forest = False
run_grid = True

prob_fire = 0.4
prob_bad_state = 0.1
prob_desired_move = 0.9

gammas = [0.1, 0.5, 0.9]
gamma_low = 0.1
gamma_med = 0.5
gamma_high = 0.9


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


def visualize_grid_world(R, X, Y):
	return(np.reshape(R, (Y, X)))


def visualize_policy(policy, dim):
	directions = {0: "v", 1: ">", 2: "^", 3: "<"}
	policy = [directions[policy_i] for policy_i in policy]
	policy_grid = np.reshape(policy, (dim, dim))
	return policy_grid


def run_gamma_sweep(mdp, vi_pi, prob_str, P, R, gammas, dim):
	if mdp is "forest":
		pass
	elif mdp is "grid":
		pass
	else:
		print("ERROR: Need forest|grid for mdp.  Passed: ", mdp)
		exit(1)
	if vi_pi is "vi":
		pass
	elif vi_pi is "pi":
		pass
	else:
		print("ERROR: Need vi|pi for vi_pi.  Passed: ", vi_pi)
		exit
	base_path = './output/csv/'+mdp+'_' + prob_str + '_'+vi_pi+'_'
	base_sweep_path = './output/'+mdp+'_' + prob_str + '_'
	gamma_sweep_file = base_sweep_path+'gamma_sweep.rpt'
	if mdp is "grid":
		gw = visualize_grid_world(R[:,0], dim, dim)
		with open(gamma_sweep_file, 'a') as f:
			f.write("Grid World is:\n"+str(gw)+"\n\n")
	for gamma in gammas:
		gamma_stats_file = base_path + 'gamma_' + str(gamma) + '.csv'
		print("Running Value Iteration with gamma", gamma)
		if vi_pi is "vi":
			alg = ValueIteration(P, R, gamma)
		elif vi_pi is "pi":
			alg = PolicyIteration(P, R, gamma)
		stats = alg.run()
		df = pd.DataFrame(stats)
		df.to_csv(gamma_stats_file, index_label="Iteration")
		print("Value Iteration complete.")
		print("Optimal value function: ", alg.V)
		print("Optimal policy: ", alg.policy)
		with open(gamma_sweep_file, 'a') as f:
			f.write("***"+vi_pi+" with Gamma="+str(gamma)+"***\n")
			if mdp is "forest":
				# Just dump policy
				f.write("Policy is:\n"+str(alg.policy)+"\n")
			if mdp is "grid":
				# Dump reshaped policy and simulated rewards
				reshaped_policy = visualize_policy(alg.policy, dim)
				simulated_rewards = get_reward(P, R, alg.policy, 10)
				f.write("Policy is:\n" + str(reshaped_policy) + "\n")
				f.write("Simulated rewards are:"+str(simulated_rewards)+"\n")
			f.write("***End of "+vi_pi+" with Gamma="+str(gamma)+"***\n\n")


if run_forest:
	for dim in forest_dims:
		P, R = forest(dim, 4, 1, 0.4, is_sparse=False)
		prob_str = str(dim)+'_'+str(prob_fire)
		run_gamma_sweep("forest", "vi", prob_str, P, R, gammas, dim)
		run_gamma_sweep("forest", "pi", prob_str, P, R, gammas, dim)

if run_grid:
	for dim in grid_dims:
		P, R = grid_world(X=dim, Y=dim, prob_desired_move=prob_desired_move, prob_bad_state=prob_bad_state)
		prob_str = str(dim) +'x'+str(dim)+ '_' + str(prob_desired_move)+'_'+str(prob_bad_state)
		run_gamma_sweep("grid", "vi", prob_str, P, R, gammas, dim)
		run_gamma_sweep("grid", "pi", prob_str, P, R, gammas, dim)


