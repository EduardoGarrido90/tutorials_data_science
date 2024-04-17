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

high_volat_port = ["AAPL","NVDA","AMD","TSLA", "QCOM"] # Aqui la que tenga volatilidad alta.
medium_volat_port =["HD","INTEL", "MMM", "XOM","HON"] # Aqui la que tenga volatilidad media.
low_volat_port = ["MCD", "KO","CL", "KMB", "PG" ] # Aqui la que tenga volatilidad baja.


df_hv = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2024-01-01',
                     ticker_list = high_volat_port).fetch_data()

df_mv = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2024-01-01',
                     ticker_list = medium_volat_port).fetch_data()

df_lv = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2024-01-01',
                     ticker_list = low_volat_port).fetch_data()

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df_hv = fe.preprocess_data(df_hv)
df_mv = fe.preprocess_data(df_mv)
df_lv = fe.preprocess_data(df_lv)


#El resto del codigo que aplique a las 3 carteras
#se debe triplicar, como por ejemplo el siguiente
#codigo. Y asi en el script.

# add covariance matrix as states
df_hv=df_hv.sort_values(['date','tic'],ignore_index=True)
df_mv=df_mv.sort_values(['date','tic'],ignore_index=True)
df_lv=df_lv.sort_values(['date','tic'],ignore_index=True)
df_hv.index = df_hv.date.factorize()[0]
df_mv.index = df_mv.date.factorize()[0]
df_lv.index = df_lv.date.factorize()[0]


cov_list_hv = []
return_list_hv = []
cov_list_mv = []
return_list_mv = []
cov_list_lv = []
return_list_lv = []

# look back is one year
#for high volatility
lookback=252
for i in range(lookback,len(df_hv.index.unique())):
  data_lookback_hv = df_hv.loc[i-lookback:i,:]
  price_lookback_hv=data_lookback_hv.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback_hv = price_lookback_hv.pct_change().dropna()
  return_list_hv.append(return_lookback_hv)

  covs_hv = return_lookback_hv.cov().values
  cov_list_hv.append(covs_hv)


df_cov_hv = pd.DataFrame({'date':df_hv.date.unique()[lookback:],'cov_list':cov_list_hv,'return_list':return_list_hv})
df_hv = df_hv.merge(df_cov_hv, on='date')
df_hv = df_hv.sort_values(['date','tic']).reset_index(drop=True)

#for medium volatility
lookback=252
for i in range(lookback,len(df_mv.index.unique())):
  data_lookback_mv = df_mv.loc[i-lookback:i,:]
  price_lookback_mv=data_lookback_mv.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback_mv = price_lookback_mv.pct_change().dropna()
  return_list_mv.append(return_lookback_mv)

  covs_mv = return_lookback_mv.cov().values
  cov_list_mv.append(covs_mv)


df_cov_mv = pd.DataFrame({'date':df_mv.date.unique()[lookback:],'cov_list':cov_list_mv,'return_list':return_list_mv})
df_mv = df_mv.merge(df_cov_mv, on='date')
df_mv = df_mv.sort_values(['date','tic']).reset_index(drop=True)

#for low volatility
lookback=252
for i in range(lookback,len(df_lv.index.unique())):
  data_lookback_lv = df_lv.loc[i-lookback:i,:]
  price_lookback_lv=data_lookback_lv.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback_lv = price_lookback_lv.pct_change().dropna()
  return_list_lv.append(return_lookback_lv)

  covs_lv = return_lookback_lv.cov().values
  cov_list_lv.append(covs_lv)


df_cov_lv = pd.DataFrame({'date':df_lv.date.unique()[lookback:],'cov_list':cov_list_lv,'return_list':return_list_lv})
df_lv = df_lv.merge(df_cov_lv, on='date')
df_lv = df_lv.sort_values(['date','tic']).reset_index(drop=True)



print(df_hv.head())
print(df_mv.head())
print(df_lv.head())

print(len(df_hv.columns))
print(len(df_mv.columns))
print(len(df_lv.columns))

import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

train_hv = data_split(df_hv, '2008-01-01','2022-07-01')
print(train_hv)
train_mv = data_split(df_mv, '2008-01-01','2022-07-01')
print(train_mv)
train_lv = data_split(df_lv, '2008-01-01','2022-07-01')
print(train_lv)

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

stock_dimension_hv = len(train_hv.tic.unique())
state_space_hv = stock_dimension_hv
print(f"Stock Dimension_hv: {stock_dimension_hv}, State Space_hv: {state_space_hv}")

stock_dimension_mv = len(train_mv.tic.unique())
state_space_mv = stock_dimension_mv
print(f"Stock Dimension_mv: {stock_dimension_mv}, State Space_hv: {state_space_mv}")

stock_dimension_lv = len(train_lv.tic.unique())
state_space_lv = stock_dimension_lv
print(f"Stock Dimension_lv: {stock_dimension_lv}, State Space_lv: {state_space_lv}")

env_kwargs_hv = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space_hv,
    "stock_dim": stock_dimension_hv,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension_hv,
    "reward_scaling": 1e-4
}

env_kwargs_mv = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space_mv,
    "stock_dim": stock_dimension_mv,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension_mv,
    "reward_scaling": 1e-4
}

env_kwargs_lv = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space_lv,
    "stock_dim": stock_dimension_lv,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension_lv,
    "reward_scaling": 1e-4
}

e_train_gym_hv = StockPortfolioEnv(df = train_hv, **env_kwargs_hv)
e_train_gym_mv = StockPortfolioEnv(df = train_mv, **env_kwargs_mv)
e_train_gym_lv = StockPortfolioEnv(df = train_lv, **env_kwargs_lv)

#env_train_hv = e_train_gym_hv.get_sb_env()
print(type(e_train_gym_hv))

#env_train_mv = e_train_gym_mv.get_sb_env()
#print(type(env_train_mv))

#env_train_lv = e_train_gym_lv.get_sb_env()
#print(type(env_train_lv))

trade_hv = data_split(df_hv,'2023-01-01', '2023-12-31')
e_trade_gym_hv = StockPortfolioEnv(df = trade_hv, **env_kwargs_hv)

trade_mv = data_split(df_mv,'2023-01-01', '2023-12-31')
e_trade_gym_mv = StockPortfolioEnv(df = trade_mv, **env_kwargs_mv)

trade_lv = data_split(df_lv,'2023-01-01', '2023-12-31')
e_trade_gym_lv = StockPortfolioEnv(df = trade_lv, **env_kwargs_lv)

import random as random
from pyfolio import timeseries
import torch

agent_hv = DRLAgent(env = e_train_gym_hv)
agent_mv = DRLAgent(env = e_train_gym_mv)
agent_lv = DRLAgent(env = e_train_gym_lv)

n_methods = 3
#seeds = np.linspace(0,24,25).astype(int) #number of experiments
seeds = np.linspace(0,4,5).astype(int) #number of experiments
results_ppo = np.zeros([n_methods, len(seeds)]) #number of methods
timesteps_for_agent = 200000 #Beta value, should be 100000, configurable for your computer, the higher the less variance and bias.
index_coefs = 0

for index_method in range(n_methods):
  PPO_PARAMS = {
      "n_steps": 2048,
      "ent_coef": 0.005,
      "learning_rate": 0.0001,
      "batch_size": 128,
  }
  model_ppo = None
  if index_method == 0:
    model_ppo = agent_hv.get_model("ppo", model_kwargs = PPO_PARAMS)
  elif index_method == 1:
    model_ppo = agent_mv.get_model("ppo", model_kwargs = PPO_PARAMS)
  else:
    model_ppo = agent_lv.get_model("ppo", model_kwargs = PPO_PARAMS)
  index_seeds = 0
  for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if index_method == 0:
      trained_ppo = model_ppo.learn(total_timesteps=timesteps_for_agent)
      df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym_hv)
      DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
      perf_func = timeseries.perf_stats
      perf_stats_all = perf_func( returns=DRL_strat,
                                factor_returns=DRL_strat,
                                  positions=None, transactions=None, turnover_denom="AGB")
      sharpe_ratio_ppo_hv = perf_stats_all["Sharpe ratio"]
      results_ppo[index_coefs, index_seeds] = sharpe_ratio_ppo_hv
      index_seeds += 1
    elif index_method == 1:
      trained_ppo = model_ppo.learn(total_timesteps=timesteps_for_agent)
      df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym_mv)
      DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
      perf_func = timeseries.perf_stats
      perf_stats_all = perf_func( returns=DRL_strat,
                                factor_returns=DRL_strat,
                                  positions=None, transactions=None, turnover_denom="AGB")
      sharpe_ratio_ppo_mv = perf_stats_all["Sharpe ratio"]
      results_ppo[index_coefs, index_seeds] = sharpe_ratio_ppo_mv
      index_seeds += 1
    else:
      trained_ppo = model_ppo.learn(total_timesteps=timesteps_for_agent)
      df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym_lv)
      DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
      perf_func = timeseries.perf_stats
      perf_stats_all = perf_func( returns=DRL_strat,
                                factor_returns=DRL_strat,
                                  positions=None, transactions=None, turnover_denom="AGB")
      sharpe_ratio_ppo_lv = perf_stats_all["Sharpe ratio"]
      results_ppo[index_coefs, index_seeds] = sharpe_ratio_ppo_lv
      index_seeds += 1

  index_coefs += 1
print(results_ppo)

plt.boxplot(results_ppo.T)  # Transpose to have methods as separate categories
plt.title('DRL comparison volatility results')
plt.xlabel('Volatility levels')
plt.ylabel('Sharpe Ratio')
plt.xticks(ticks=range(1, results_ppo.shape[0] + 1), labels=['High','Medium','Low'])
plt.savefig('results.png')
