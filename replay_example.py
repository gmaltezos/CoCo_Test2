from py_agata.py_agata import Agata
from py_agata.time_in_ranges import time_in_target
from py_agata.utils import glucose_time_vectors_to_dataframe

from plot_results import plot_results

from replay_bg_wrapper import simulate
from basal_insulin_controller_example import basal_insulin_controller_example
from basal_insulin_controller_mpc import MPC
from basal_insulin_controller_deepc import DeePC
from basal_insulin_controller_pid import PID
import pandas as pd
import os

# You can create a dict() that will be accessible during simulation by your handler
basal_handler_params = dict()
basal_handler_params['default_rate'] = 0.01

# This is the call to the wrapper I prepared. You need to:
#      - Set meal_input_modulation_factor <> 1 if you want to increase/decrease the meal input (=1 means the original meal input)
#      - pass the function to manage basal insulin (i.e., basal_handler)
#      - optionally pass parameters that you want to be used by your handler (i.e., basal_handler_params)
# The wrapper will return arrays that represent the simulated glucose, insulin_bolus, cho(m), insulin_basal(i), and time.
# Remember that time and cho will be always the same, while insulin_bolus is calculated using
# a standard formula of meal insulin bolus calculation than depends on cho amount and current glucose level.
# As such, you will control only insulin_basal.
#Note that you will simulate ~30days. I generated a scenario that is not super physiological but is super ok for teaching!
# Get test data
data_given = pd.read_csv(
        os.path.join(os.path.abspath(''), 'data', 'data_cho_pid.csv'))

glucose, i, insulin_bolus, m, time = simulate(basal_handler=DeePC,
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
plot_results(glucose=glucose,insulin_bolus=insulin_bolus, i=i, m=m, time=time)
