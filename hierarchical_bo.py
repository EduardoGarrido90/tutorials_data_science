import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
import sys
sys.path.append("../FinRL-Library")


import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


# <a id='2'></a>
# # Part 3. Download Data
# Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** to fetch data from Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
# 

# In[ ]:


print(config_tickers.DOW_30_TICKER)


# In[ ]:


# Download and save the data in a pandas DataFrame:
#Ojo, esto hay que cambiarlo para que nos adaptemos a un nuevo mercado y fechas.
#Podemos usar factset en vez de Yahoo, que va mucho mejor.
df = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2023-12-31',
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()


# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in highly volatile markets, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation in order to XXXXX.

# In[ ]:


#Aquí también se pueden introducir mejoras.
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df = fe.preprocess_data(df)


# ## Add covariance matrix as states

# In[ ]:


# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  return_list.append(return_lookback)

  covs = return_lookback.cov().values
  cov_list.append(covs)


df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)


# <a id='4'></a>
# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
# 
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
# 

# In[ ]:


import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


# ## Training data split: 2008-01-01 to 2022-12-31
# Se debería hacer 10 Fold CV temporal para mejorar a nivel de empresa.

# In[ ]:


train = data_split(df, '2008-01-01','2022-12-31')


# Here is the definition of the environment.

# In[ ]:


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                lookback=252,
                day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators + ESG (4). Ojo, no funciona meter aqui el shape bueno. Esto puede causar problemas.

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list), self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]

        self.state = np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)


        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]


    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(),'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()

            plt.plot(self.portfolio_return_memory,'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal,{}

        else:
            #print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            #if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            #else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            #print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            #print(self.state)

            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)

            #...Weights tbc by investor´s preference
            # portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)

            #Aqui es donde hay que ponderar el ESG.
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            #print("Step reward: ", self.reward)
            #self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)

        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]
        return self.state

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output


    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


# In[ ]:


stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[ ]:


env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)


# In[ ]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# Bayesian optimization baseline and random search blackbox

# In[ ]:

from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from pyfolio import timeseries

#Vanilla Bayesian optimization baseline.
#We need to include everything from agent name to hyperparameters.
#Obviously the search in such a space won't work well.
#We will use as target the Sharpe ratio.
#def financial_portfolio_optimization_baseline(ent_coef, learning_rate, gamma, clip_range, gae_lambda, tau, ppo_ddpg, seed: int = 0) -> float:
def financial_portfolio_optimization_baseline(config: Configuration, seed: int = 0) -> float:
    timesteps = 100000 #Beta value for timesteps. For the experiments, needs to be higher (+-80000)
    agent = DRLAgent(env = env_train)
    agent_name = None
    PARAMS = None
    if config["algorithm"] == 0:
      agent_name = "ppo"
      PARAMS = {
        "n_steps": 2048,
        "ent_coef": config["ent_coef"],
        "learning_rate": config["learning_rate"],
        "batch_size": 128,
        "gamma" : config["gamma"],
        "clip_range" : config["clip_range"],
        "gae_lambda" : config["gae_lambda"]
      }
    else:
      agent_name = "ddpg"
      PARAMS = {
        "learning_rate": config["learning_rate"],
        "batch_size": 128,
        "gamma" : config["gamma"],
        "tau" : config["tau"]
      }

    model = agent.get_model(agent_name, model_kwargs = PARAMS)
    trained_agent = agent.train_model(model=model,
                             tb_log_name=agent_name,
                             total_timesteps=timesteps)

    trade = data_split(df,'2023-01-01', '2023-12-31')
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_agent, environment = e_trade_gym)

    DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
    return perf_stats_all["Sharpe ratio"]



def financial_portfolio_optimization_rs(ent_coef, learning_rate, gamma, clip_range, gae_lambda, tau, algorithm) -> float:
    timesteps = 100000 #Beta value for timesteps. For the experiments, needs to be higher (+-80000)
    agent = DRLAgent(env = env_train)
    agent_name = None
    PARAMS = None
    if algorithm == 0:
      agent_name = "ppo"
      PARAMS = {
        "n_steps": 2048,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "batch_size": 128,
        "gamma" : gamma,
        "clip_range" : clip_range,
        "gae_lambda" : gae_lambda
      }
    else:
      agent_name = "ddpg"
      PARAMS = {
        "learning_rate": learning_rate,
        "batch_size": 128,
        "gamma" : gamma,
        "tau" : tau
      }

    model = agent.get_model(agent_name, model_kwargs = PARAMS)
    trained_agent = agent.train_model(model=model,
                             tb_log_name=agent_name,
                             total_timesteps=timesteps)

    trade = data_split(df,'2023-01-01', '2023-12-31')
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_agent, environment = e_trade_gym)

    DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
    return perf_stats_all["Sharpe ratio"]



# Hierarchical Bayesian optimization

# In[ ]:


#def financial_portfolio_hierarchical_optimization(ent_coef, learning_rate, gamma, clip_range, gae_lambda):
def financial_portfolio_hierarchical_optimization(config: Configuration, seed: int = 0) -> float:
    timesteps = 100000 #Beta value for timesteps. For the experiments, needs to be higher (+-80000)
    agent = DRLAgent(env = env_train)
    agent_name = None
    PARAMS = None
    if config["algorithm"] == 0:
      agent_name = "ppo"
      PARAMS = {
        "n_steps": 2048,
        "ent_coef": config["ent_coef"],
        "learning_rate": config["learning_rate_ppo"],
        "batch_size": 128,
        "gamma" : config["gamma_ppo"],
        "clip_range" : config["clip_range"],
        "gae_lambda" : config["gae_lambda"]
      }
    else:
      agent_name = "ddpg"
      PARAMS = {
        "learning_rate": config["learning_rate_ddpg"],
        "batch_size": 128,
        "gamma" : config["gamma_ddpg"],
        "tau" : config["tau"]
      }

    model_ppo = agent.get_model(agent_name, model_kwargs = PARAMS)

    trained_ppo = agent.train_model(model=model_ppo,
                             tb_log_name=agent_name,
                             total_timesteps=timesteps)

    trade = data_split(df,'2023-01-01', '2023-12-31')
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym)

    DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
    return perf_stats_all["Sharpe ratio"]


# Experiment launcher

# In[ ]:


from smac import HyperparameterOptimizationFacade, Scenario
import random as random
from scipy.optimize import NonlinearConstraint
from ConfigSpace.conditions import EqualsCondition

number_repetitions = 3 #Must be 25 for real experiments.
seeds = np.linspace(0, number_repetitions-1, number_repetitions).astype(int)
methods = {"HBO": 0, "BO_BASELINE": 1, "RS" : 2}
n_iters = 20 #Must be 50 in real experiments.
results_experiment = np.zeros([len(methods.keys()), len(seeds), n_iters]) #Methods, experiments, iterations.

ppo_hp_bounds = {'ent_coef': (0.0, 0.1),
                 'learning_rate': (0.000001, 0.1),
                 "gamma" : (0.9, 0.9999),
                 "clip_range": (0.1, 0.3),
                 "gae_lambda" : (0.9, 0.999)}

ddpg_hp_bounds = {'tau': (0.0, 0.1),
                 'learning_rate': (0.000001, 0.1),
                 "gamma" : (0.9, 0.9999)}

for seed in seeds:

  #Hierarchical Bayesian optimization.
  # Define the configuration space
  configspace = ConfigurationSpace(seed=seed)

  # Choose between algorithms (0:ppo, 1:ddpg) could be more....
  algorithm = CategoricalHyperparameter('algorithm', [0, 1])
  configspace.add_hyperparameter(algorithm)

  # PPO Hyperparameters
  ent_coef = UniformFloatHyperparameter('ent_coef', 0.0, 0.1)
  learning_rate_ppo = UniformFloatHyperparameter('learning_rate_ppo', 0.000001, 0.1)
  gamma_ppo = UniformFloatHyperparameter('gamma_ppo', 0.9, 0.9999)
  clip_range = UniformFloatHyperparameter('clip_range', 0.1, 0.3)
  gae_lambda = UniformFloatHyperparameter('gae_lambda', 0.9, 0.999)

  # DDPG Hyperparameters
  learning_rate_ddpg = UniformFloatHyperparameter('learning_rate_ddpg', 0.000001, 0.1, default_value=0.001)
  gamma_ddpg = UniformFloatHyperparameter('gamma_ddpg', 0.9, 0.9999, default_value=0.99)
  tau = UniformFloatHyperparameter('tau', 0.0, 0.1, default_value=0.01)

  # Add PPO hyperparameters
  configspace.add_hyperparameters([ent_coef, learning_rate_ppo, gamma_ppo, clip_range, gae_lambda])

  # Add DDPG hyperparameters
  configspace.add_hyperparameters([learning_rate_ddpg, gamma_ddpg, tau])

  # Define conditions
  configspace.add_condition(EqualsCondition(ent_coef, algorithm, 0))
  configspace.add_condition(EqualsCondition(learning_rate_ppo, algorithm, 0))
  configspace.add_condition(EqualsCondition(gamma_ppo, algorithm, 0))
  configspace.add_condition(EqualsCondition(clip_range, algorithm, 0))
  configspace.add_condition(EqualsCondition(gae_lambda, algorithm, 0))

  configspace.add_condition(EqualsCondition(learning_rate_ddpg, algorithm, 1))
  configspace.add_condition(EqualsCondition(gamma_ddpg, algorithm, 1))
  configspace.add_condition(EqualsCondition(tau, algorithm, 1))

  # Scenario object specifying the optimization environment
  scenario = Scenario(configspace, deterministic=True, n_trials=n_iters)

  # Use SMAC to find the best configuration/hyperparameters
  smac = HyperparameterOptimizationFacade(scenario, financial_portfolio_hierarchical_optimization)
  incumbent = smac.optimize()
  sharpe = financial_portfolio_hierarchical_optimization(incumbent)

  results_experiment[methods["HBO"], seed] = np.array([sharpe for i in range(n_iters)])

  #Baseline Bayesian optimization.
  # Define the configuration space
  configspace_2 = ConfigurationSpace(seed=seed)

  # Add float hyperparameters
  configspace_2.add_hyperparameters([
    UniformFloatHyperparameter('ent_coef', 0.0, 0.1),
    UniformFloatHyperparameter('learning_rate', 0.000001, 0.1),
    UniformFloatHyperparameter('gamma', 0.9, 0.9999),
    UniformFloatHyperparameter('clip_range', 0.1, 0.3),
    UniformFloatHyperparameter('gae_lambda', 0.9, 0.999),
    UniformFloatHyperparameter('tau', 0.0, 0.1)
  ])

  # Add integer hyperparameter as a categorical since it has discrete values
  configspace_2.add_hyperparameter(CategoricalHyperparameter('algorithm', [0, 1]))


  # Scenario object specifying the optimization environment
  scenario = Scenario(configspace_2, deterministic=True, n_trials=n_iters)

  # Use SMAC to find the best configuration/hyperparameters
  smac_2 = HyperparameterOptimizationFacade(scenario, financial_portfolio_optimization_baseline)
  incumbent = smac_2.optimize()
  sharpe = financial_portfolio_optimization_baseline(incumbent)

  results_experiment[methods["BO_BASELINE"], seed] = np.array([sharpe for i in range(n_iters)])

  #Random search.
  random.seed(seed)
  for iter in range(n_iters):
    ent_coef_random = random.uniform(0.0, 0.1)
    learning_rate_random = random.uniform(0.000001, 0.1)
    gamma_random = random.uniform(0.9, 0.9999)
    clip_range_random = random.uniform(0.1, 0.3)
    gae_lambda = random.uniform(0.9, 0.999)
    tau_random = random.uniform(0.0, 0.1)
    ppo_ddpg = random.randint(0, 1) #0: ppo, 1:ddpg
    results_experiment[methods["RS"], seed, iter] = financial_portfolio_optimization_rs(ent_coef_random, learning_rate_random, gamma_random, clip_range_random, gae_lambda, tau_random, ppo_ddpg)

  print("Iteration #" + str(seed) + " done.")

#3. Print the results:
#array structure: results_experiment = np.zeros([len(methods.keys()), len(seeds), n_iters]) #Methods, experiments, iterations.
#means = np.mean(results_experiment, axis=1)[n_iters]
#stds = np.std(results_experiment, axis=1)[n_iters]

# Extract the last iteration results for each method across all seeds
last_iter_results = results_experiment[:, :, -1] # Plot instead the maximum of iters.

# Plotting
fig, ax = plt.subplots()
method_names = list(methods.keys())
# Create boxplots for each method
ax.boxplot(last_iter_results.T, labels=method_names)  # Transpose for correct orientation
ax.set_title('Results for last iteration across all Seeds')
ax.set_ylabel('Result')
ax.set_xlabel('Method')
plt.xticks(rotation=45)  # Rotate method names for better visibility
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.savefig("results.png")
