import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_convergence(df, attr, title, xlabel, ylabel, file, cum=False):
    plt.figure()
    plt.title(title)
    if cum:
        y = df[attr].cumsum()
    else:
        y = df[attr]
    plt.plot(df['Iteration'], y, 'o-', color='g')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file)



def get_data(file):
    df = pd.read_csv(file)
    return df


def get_forest_data(num_states, prob_fire):
    ql_df = get_data('output/csv/forest_'+str(num_states)+'_'+str(prob_fire)+'_stats_ql.csv')
    vi_df = get_data('output/csv/forest_'+str(num_states)+'_'+str(prob_fire)+'_stats_vi.csv')
    pi_df = get_data('output/csv/forest_'+str(num_states)+'_'+str(prob_fire)+'_stats_pi.csv')
    return pi_df, vi_df, ql_df


def get_grid_data(dim, prob_desired_move, prob_bad_state):
    ql_df = get_data('output/csv/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired_move)+'_'+str(prob_bad_state)+'_ql.csv')
    vi_df = get_data('output/csv/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired_move)+'_'+str(prob_bad_state)+'_vi.csv')
    pi_df = get_data('output/csv/grid_'+str(dim)+'x'+str(dim)+'_'+str(prob_desired_move)+'_'+str(prob_bad_state)+'_pi.csv')
    return pi_df, vi_df, ql_df


def plot_forest_convergence_plots(num_states, prob_fire):
    pi_df, vi_df, ql_df = get_forest_data(num_states, prob_fire)
    plot_convergence(vi_df, 'Error', "Forest "+str(num_states)+","+str(prob_fire)+" VI Convergence",
                     xlabel="Iteration Number", ylabel="Error",
                     file='output/plot/Forest_'+str(num_states)+'_'+str(prob_fire)+'_MDP_VI_Convergence.png')
    plot_convergence(pi_df, 'Error', "Forest "+str(num_states)+","+str(prob_fire)+" PI Convergence",
                     xlabel="Iteration Number", ylabel="Error",
                     file='output/plot/Forest_'+str(num_states)+'_'+str(prob_fire)+'_MDP_PI_Convergence.png')
    plot_convergence(ql_df, 'Mean V', "Forest "+str(num_states)+","+str(prob_fire)+" QLearning Convergence",
                     xlabel="Iteration Number (10e3)", ylabel="Mean Value",
                     file='output/plot/Forest_'+str(num_states)+'_'+str(prob_fire)+'_MDP_QL_Convergence.png')


def plot_grid_convergence_plots(dim, prob_desired_move, prob_bad_state):
    # Get data
    pi_df, vi_df, ql_df = get_grid_data(dim, prob_desired_move, prob_bad_state)

    # Build strs
    prob_str = str(dim)+'x'+str(dim)+','+str(prob_desired_move)+','+str(prob_bad_state)
    file_prob_str = str(dim)+'x'+str(dim)+'_'+str(prob_desired_move)+'_'+str(prob_bad_state)

    # Plot
    plot_convergence(vi_df, 'Error', "Grid "+prob_str+" VI Convergence",
                     xlabel="Iteration Number", ylabel="Error",
                     file='output/plot/Grid_'+file_prob_str+'_MDP_VI_Convergence.png')
    plot_convergence(pi_df, 'Error', "Grid "+prob_str+" PI Convergence",
                     xlabel="Iteration Number", ylabel="Error",
                     file='output/plot/Grid_'+file_prob_str+'_MDP_PI_Convergence.png')
    plot_convergence(ql_df, 'Mean V', "Grid "+prob_str+" QLearning Convergence",
                     xlabel="Iteration Number", ylabel="Mean Value",
                     file='output/plot/Grid_'+file_prob_str+'_MDP_QL_Convergence.png')


def get_learning_curve_data_forest(param, num_states, prob_fire, value, decay):
    prob_str = str(num_states)+'_'+str(prob_fire)
    base_path = './output/csv/forest_'+prob_str+'_'+param+'_'+str(value)+'_'
    if decay:
        path = base_path+'decay.csv'
    else:
        path = base_path+'no_decay.csv'
    df = get_data(path)
    return df


def get_learning_curve_data_grid(param, dim, prob_move, prob_bad, value, decay):
    prob_str = str(dim)+"x"+str(dim)+'_'+str(prob_move)+'_'+str(prob_bad)
    base_path = './output/csv/grid_'+prob_str+'_'+param+'_'+str(value)+'_'
    if decay:
        path = base_path+'decay.csv'
    else:
        path = base_path+'no_decay.csv'
    df = get_data(path)
    return df


def get_gamma_data_grid(param, dim, prob_move, prob_bad, value):
    prob_str = str(dim)+"x"+str(dim)+'_'+str(prob_move)+'_'+str(prob_bad)
    path = './output/csv/grid_'+prob_str+'_'+param+'_'+str(value)+'.csv'
    df = get_data(path)
    return df


def get_gamma_data_forest(param, num_states, prob_fire, value):
    prob_str = str(num_states)+'_'+str(prob_fire)
    path = './output/csv/forest_'+prob_str+'_'+param+'_'+str(value)+'.csv'
    df = get_data(path)
    return df

def plot_learning_curve(mdp, param, title, file):
    # Get data
    if mdp is "grid":
        if param is "alpha":
            df_low = get_learning_curve_data_grid(param, 10, 0.9, 0.1, 0.001, False)
            df_high = get_learning_curve_data_grid(param, 10, 0.9, 0.1, 0.5, False)
            df_decay = get_learning_curve_data_grid(param, 10, 0.9, 0.1, 0.1, True)
            low_str = "Alpha = 0.001"
            high_str = "Alpha = 0.5"
            decay_str = "Alpha = 0.1 with Decay"
        elif param is "epsilon":
            df_low = get_learning_curve_data_grid(param, 10, 0.9, 0.1, 0.1, False)
            df_high = get_learning_curve_data_grid(param, 10, 0.9, 0.1, 0.9, False)
            df_decay = get_learning_curve_data_grid(param, 10, 0.9, 0.1, 1.0, True)
            low_str = "Epsilon = 0.1"
            high_str = "Epsilon = 0.9"
            decay_str = "Epsilon = 1.0 with Decay"
    elif mdp is "forest":
        if param is "alpha":
            df_low = get_learning_curve_data_forest(param, 10, 0.4, 0.001, False)
            df_high = get_learning_curve_data_forest(param, 10, 0.4, 0.5, False)
            df_decay = get_learning_curve_data_forest(param, 10, 0.4, 0.1, True)
            low_str = "Alpha = 0.001"
            high_str = "Alpha = 0.5"
            decay_str = "Alpha = 0.1 with Decay"
        elif param is "epsilon":
            df_low = get_learning_curve_data_forest(param, 10, 0.4, 0.1, False)
            df_high = get_learning_curve_data_forest(param, 10, 0.4, 0.9, False)
            df_decay = get_learning_curve_data_forest(param, 10, 0.4, 1.0, True)
            low_str = "Epsilon = 0.1"
            high_str = "Epsilon = 0.9"
            decay_str = "Epsilon = 1.0 with Decay"
        else:
            print("ERROR: MDP is ", mdp, " which is not a valid mdp type.  Select forest or grid")
            exit
    attr = 'Mean V'
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.plot(df_low['Iteration'], df_low[attr], 'o-', color='r', label=low_str)
    plt.plot(df_high['Iteration'], df_high[attr], 's-', color='b', label=high_str)
    plt.plot(df_decay['Iteration'], df_decay[attr], 'D--', color='g', label=decay_str)
    plt.xlabel("Iteration (10e3)")
    plt.ylabel("Mean Value")
    plt.legend(loc="best")
    plt.savefig(file)


def plot_gamma_curve(mdp, title, file):
    # Get data
    param = "gamma"
    if mdp is "grid":
            df_low = get_gamma_data_grid(param, 10, 0.9, 0.1, 0.001)
            df_med = get_gamma_data_grid(param, 10, 0.9, 0.1, 0.5)
            df_high = get_gamma_data_grid(param, 10, 0.9, 0.1, 0.1)
    elif mdp is "forest":
            df_low = get_gamma_data_forest(param, 10, 0.4, 0.1)
            df_med = get_gamma_data_forest(param, 10, 0.4, 0.5)
            df_high = get_gamma_data_forest(param, 10, 0.4, 0.9)
    else:
        print("ERROR: MDP is ", mdp, " which is not a valid mdp type.  Select forest or grid")
        exit
    low_str = "Gamma = 0.1"
    med_str = "Gamma = 0.5"
    high_str = "Gamma = 0.9"
    attr = 'Mean V'
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.plot(df_low['Iteration'], df_low[attr], 'o-', color='r', label=low_str)
    plt.plot(df_med['Iteration'], df_med[attr], 's-', color='b', label=med_str)
    plt.plot(df_high['Iteration'], df_high[attr], 'D--', color='g', label=high_str)
    plt.xlabel("Iteration (10e3)")
    plt.ylabel("Mean Value")
    plt.legend(loc="best")
    plt.savefig(file)


# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def plot_grid_world_policy(policy, rewards, dim, title, file):
    fig, ax = plt.subplots()

    # Color by rewards
    im = ax.imshow(rewards)

    # Labels
    ax.set_xticks(np.arange(dim+1)-0.5)
    ax.set_yticks(np.arange(dim+1)-0.5)
    ax.set_xticklabels(range(dim))
    ax.set_yticklabels(range(dim))

    # Annotate policy
    for i in range(dim):
        for j in range(dim):
            text = ax.text(j, i, policy[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    #fig.tight_layout()
    plt.savefig(file)


def plot_sample_optimal_policies():
    world = np.array([[-1,  -1,  -1],
                      [-1,  -1,  -1],
                      [-1,  -1, 100]])
    optimal_policy_1 = np.array([['>', 'v', 'v'],
                                ['>', '>', 'v'],
                                ['>', '>', 'v']])
    optimal_policy_2 = np.array([['v', 'v', 'v'],
                                 ['v', '>', 'v'],
                                 ['>', '>', 'v']])
    plot_grid_world_policy(world, world, 3, "3x3 Grid World Rewards", "./output/plot/3x3_grid_world.png")
    plot_grid_world_policy(optimal_policy_1, world, 3, "Sample Optimal Policy", "./output/plot/3x3_optimal_policy1.png")
    plot_grid_world_policy(optimal_policy_2, world, 3, "Sample Optimal Policy", "./output/plot/3x3_optimal_policy2.png")

#plot_forest_convergence_plots(3, 0.4)
#plot_forest_convergence_plots(10, 0.4)
#plot_forest_convergence_plots(10, 0.1)
plot_forest_convergence_plots(400, 0.4)
#plot_grid_convergence_plots(3, 0.9, 0.1)

# Grid
#plot_learning_curve("grid", "alpha", "Impact of Alpha on Learning", './output/plot/alpha_grid_lc_10x10.png')
#plot_learning_curve("grid", "epsilon", "Impact of Epsilon on Learning", './output/plot/epsilon_grid_lc_10x10.png')

# Forest
#plot_learning_curve("forest", "alpha", "Impact of Alpha on Learning", './output/plot/alpha_forest_lc_10x10.png')
#plot_learning_curve("forest", "epsilon", "Impact of Epsilon on Learning", './output/plot/epsilon_forest_lc_10x10.png')
#plot_gamma_curve("forest", "Impact of Gamma on Learning", './output/plot/gamma_forest_lc_10x10.png')

#plot_grid_convergence_plots(10, 0.9, 0.1)

# Rewards
world = np.array([[-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -20],
 [-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [-1, -20,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [-1,  -1,  -1,  -1,  -1,  -1, -20,  -1,  -1,  -1],
 [-1,  -1,  -1,  -1, -20,  -1,  -1, -20,  -1,  -1],
 [-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [-1,  -1,  -1,  -1, -20,  -1,  -1, -20,  -1,  -1],
 [-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [-1,  -1,  -1, -20,  -1,  -1,  -1, -20,  -1,  -1],
 [-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 100]])
# Policy
# Gamma 0.1
policy_gamma0p1 = np.array([['v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', '<', 'v'],
['v', '^', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'],
['<', 'v', '>', 'v', 'v', 'v', '^', '^', 'v', 'v'],
['v', 'v', 'v', 'v', '^', '<', '^', '^', '>', 'v'],
['v', 'v', 'v', '<', '^', '>', 'v', '>', '>', 'v'],
['v', 'v', 'v', '<', '>', '>', '<', '>', '>', 'v'],
['v', 'v', 'v', '<', 'v', '>', '<', '>', '>', 'v'],
['v', 'v', 'v', '^', 'v', 'v', '<', '>', 'v', 'v'],
['v', 'v', '<', 'v', '>', 'v', '<', '>', '>', 'v'],
['v', 'v', 'v', 'v', 'v', 'v', '>', '>', '>', 'v']])

# Gamma 0.5
policy_gamma0p5 = np.array([['v', 'v', 'v', 'v', 'v', '>', '>', 'v', '<', 'v'],
['v', '^', 'v', 'v', '>', '>', '>', '>', 'v', 'v'],
['<', 'v', '>', '>', '>', '>', '^', '>', 'v', 'v'],
['v', 'v', 'v', '<', '^', '<', '^', '>', 'v', 'v'],
['v', 'v', 'v', '<', '^', '>', 'v', '>', '>', 'v'],
['v', 'v', 'v', '<', '>', '>', 'v', '>', 'v', 'v'],
['v', 'v', 'v', '<', 'v', '>', 'v', '>', '>', 'v'],
['v', 'v', 'v', '^', 'v', 'v', '>', '>', 'v', 'v'],
['v', 'v', '<', '>', '>', 'v', 'v', '>', '>', 'v'],
['>', '>', '>', '>', '>', '>', '>', '>', '>', 'v']])

# Gamma 0.9
policy_gamma0p9 = np.array([['>', '>', '>', '>', '>', '>', '>', 'v', 'v', 'v'],
['v', '>', '>', '>', '>', '>', '>', 'v', 'v', 'v'],
['v', 'v', '>', '>', '>', 'v', '>', '>', 'v', 'v'],
['v', 'v', 'v', 'v', '>', 'v', 'v', '>', 'v', 'v'],
['v', 'v', 'v', 'v', '>', '>', 'v', '>', 'v', 'v'],
['v', 'v', '>', '>', '>', '>', 'v', '>', 'v', 'v'],
['v', '>', 'v', 'v', '>', '>', 'v', '>', 'v', 'v'],
['>', '>', '>', '>', '>', '>', '>', '>', 'v', 'v'],
['>', '>', 'v', '>', '>', '>', 'v', '>', '>', 'v'],
['>', '>', '>', '>', '>', '>', '>', '>', '>', 'v']])

policy_QL = np.array([['>', '>', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v'],
['>', '>', '>', '>', '>', '>', '>', 'v', 'v', 'v'],
['v', 'v', '>', '>', '>', '>', '>', '>', 'v', 'v'],
['v', 'v', 'v', 'v', '^', 'v', '>', '>', '>', 'v'],
['>', 'v', 'v', 'v', '>', 'v', 'v', '>', '>', 'v'],
['>', '>', 'v', 'v', '>', 'v', '>', '>', '>', 'v'],
['>', '>', 'v', 'v', 'v', 'v', 'v', '>', '>', 'v'],
['>', '>', '>', '>', 'v', 'v', 'v', '>', '>', 'v'],
['>', 'v', 'v', '>', '>', 'v', 'v', '>', '>', 'v'],
['>', '>', '>', '>', '>', '>', '>', '>', '>', 'v']])

policy_PI = np.array([['>', '>', '>', '>', '>', '>', '>', 'v', 'v', 'v'],
['v', '>', '>', '>', '>', '>', '>', 'v', 'v', 'v'],
['v', 'v', '>', '>', '>', 'v', '>', '>', 'v', 'v'],
['v', 'v', 'v', 'v', '>', 'v', 'v', '>', 'v', 'v'],
['v', 'v', 'v', 'v', '>', '>', 'v', '>', 'v', 'v'],
['v', 'v', '>', '>', '>', '>', 'v', '>', 'v', 'v'],
['v', '>', 'v', 'v', '>', '>', 'v', '>', 'v', 'v'],
['>', '>', '>', '>', '>', '>', '>', '>', 'v', 'v'],
['>', '>', 'v', '>', '>', '>', 'v', '>', '>', 'v'],
['>', '>', '>', '>', '>', '>', '>', '>', '>', 'v']])

policy_VI = np.array([['>', '>', '>', '>', '>', '>', '>', 'v', 'v', 'v'],
['v', '>', '>', '>', '>', '>', '>', 'v', 'v', 'v'],
['v', 'v', '>', '>', '>', 'v', '>', '>', 'v', 'v'],
['v', 'v', 'v', 'v', '>', 'v', 'v', '>', 'v', 'v'],
['v', 'v', 'v', 'v', '>', '>', 'v', '>', 'v', 'v'],
['v', 'v', '>', '>', '>', '>', 'v', '>', 'v', 'v'],
['v', '>', 'v', 'v', '>', '>', 'v', '>', 'v', 'v'],
['>', '>', '>', '>', '>', '>', '>', '>', 'v', 'v'],
['>', '>', 'v', '>', '>', '>', 'v', '>', '>', 'v'],
['>', '>', '>', '>', '>', '>', '>', '>', '>', 'v']])

#plot_grid_world_policy(world, world, 10, "10x10 Grid World Rewards", "./output/plot/10x10_grid_world.png")
#plot_grid_world_policy(policy_gamma0p1, world, 10, "VI Policy with gamma = 0.1", "./output/plot/vi_policy_gamma_0p1.png")
#plot_grid_world_policy(policy_gamma0p5, world, 10, "VI Policy with gamma = 0.5", "./output/plot/vi_policy_gamma_0p5.png")
#plot_grid_world_policy(policy_gamma0p9, world, 10, "VI Policy with gamma = 0.9", "./output/plot/vi_policy_gamma_0p9.png")
#plot_grid_world_policy(policy_QL, world, 10, "Q Learning Policy", "./output/plot/ql_policy_10x10.png")
#plot_grid_world_policy(policy_PI, world, 10, "Policy Iteration Policy", "./output/plot/pi_policy_10x10.png")
#plot_grid_world_policy(policy_VI, world, 10, "Value Iteration Policy", "./output/plot/vi_policy_10x10.png")

#plot_sample_optimal_policies()


