from py_agata.py_agata import Agata
from py_agata.time_in_ranges import time_in_target
from py_agata.utils import glucose_time_vectors_to_dataframe

from plot_results import plot_results

from replay_bg_wrapper import simulate
from basal_insulin_handler_example import basal_insulin_handler_example
# from basal_insulin_handler_mpc import basal_insulin_handler_mpc
# from basal_insulin_handler_mpc2 import basal_insulin_handler_mpc2
from basal_insulin_handler_mpc_clean import basal_insulin_handler_clean
from basal_insulin_handler_pid import basal_insulin_handler_pid
import pandas as pd
import os

# You can create a dict() that will be accessible during simulation by your handler
basal_handler_params = dict()
basal_handler_params['default_rate'] = 0.01

data_given = pd.read_csv(
        os.path.join(os.path.abspath(''), 'data', 'data_cho_pid.csv'))

# Paramters
number_of_steps = 1000
total_reward = 0.0 # Total reward


for i in range(number_of_steps):

        glucose, insulin_basal, insulin_bolus, cho, time = simulate(basal_handler=basal_insulin_handler_clean,
        basal_handler_params=basal_handler_params, data_given=data_given,
        meal_input_modulation_factor=1)

# After simulation, you can evaluate glucose trace with AGATA.
# First you need to generate a dataframe that is compatible with AGATA
data = glucose_time_vectors_to_dataframe(glucose=glucose, t=time)

# Thenm glucose trace can be analyzed with Agata().analyze_glucose_profile() if you want to obtain an exhaustive set of metrics...
agata = Agata()
results = agata.analyze_glucose_profile(data)
print(results)

# ... or, if you need just 1 metric, using single targeted functions (e.g., time spent in teh target range)
tit = time_in_target(data)
print('TIT = ' + str(tit) + '%')

#Finally, I prepared a function to plot the simulation traces
plot_results(glucose=glucose,insulin_bolus=insulin_bolus, insulin_basal=insulin_basal, cho=cho, time=time)
