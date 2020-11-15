import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from sklearn.linear_model import LinearRegression
from datetime import datetime as dt
import jinja2


def acceleration_func(t, MSS, tau):
    """Small function used when fitting curve to each sprint effort"""
    return MSS * (1 - np.e**(-t/tau))


class fvProfile:

    def __init__(self, data, name, bw):

        self.df = data
        self.sprints = []
        self.sprint_dict = {}
        self.lowess = sm.nonparametric.lowess
        self.bw = bw 
        self.name = name

    def smooth_data(self, velocity_col='Velocity', time_col='Seconds', span=0.05):
        """Using statmodels LOWESS function to create smoothed locally weighted
        linear regression"""
        self.df.rename(columns={velocity_col:'vel', time_col:'time'}, inplace=True)
        smooth_velocity = self.lowess(self.df['vel'],
                  self.df['time'],
                  frac=span)
        self.df['smooth_velocity'] = smooth_velocity[:,1]
        #plt.plot(self.df['time'], self.df['vel'])
        #plt.plot(self.df['time'], self.df['smooth_velocity'])

    def find_peaks(self, rel_height=0.4):
        """Using scipy signal module to find and extract peaks"""
        min_peak_v = self.df['smooth_velocity'].max() * 0.7

        peaks = find_peaks(self.df['smooth_velocity'], height=min_peak_v)
        peak_size = peak_widths(self.df['smooth_velocity'], peaks[0], 0.90)
        self.peaks = peaks
        self.peak_details = peak_size

    def extract_peaks(self):
        """Takes the peak data and creates a list of dataframes containing
        each individual sprint effort"""
        for i in range(len(self.peaks[0])):
            left, right = int(self.peak_details[2][i]), int(self.peak_details[3][i])
            peak = self.df.iloc[left:right]
            self.sprints.append(peak)



    def extract_accel_phase(self):
        """Takes each peak in the peak_list (list of sprint efforts), finds a start point and
        plateau point in order to extract only the accel phase"""
        for i, sprint in enumerate(self.sprints):
            decel = sprint['vel'].max() - 0.4
            # Work from end of sprint and find first point where the velocity is less than
            end_index = np.argmax(sprint['vel'][::-1] > decel)
            # Find start velcoity with threshold of 1.5m/s
            moving_index = np.argmax(sprint['vel'] > 1.5)
            # Find the first zero velocity point back from the moving index
            start_index = np.argmin(sprint['vel'][moving_index:0:-1] > 0.5)
            """NEED TEST CONDITION IF NO STATIC START"""
            # Find the point of first movement where velocity > 0.3 and add to start_index
            zero_index = moving_index - start_index
            self.sprints[i] = sprint.iloc[zero_index:-end_index]



    def model_sprint_data(self):
        """Using method from R Shorts paackage, fit curve to estimate MSS and tau"""
        for i, sprint in enumerate(self.sprints):
            self.sprints[i]['time'] = self.sprints[i]['time'] - self.sprints[i]['time'].min()
            # Fit curve to estimate MSS and tau. Need to look at bounding values
            (MSS, tau), _ = curve_fit(acceleration_func, sprint['time'], sprint['vel'], bounds=([1.,0.5], [9.,3.]))
            # Calc max acceleration
            MAC = MSS/tau
            # Calc relative max power
            PMAX = (MSS * MAC) / 4
            predicted_velocity = acceleration_func(sprint['time'], MSS, tau)

            data = {'MSS': MSS,
                   'tau': tau,
                   'MAC': MAC,
                   'PMAX': PMAX}

            self.sprint_dict[f'sprint_{i}'] = data
            self.sprints[i]['predicted_velocity'] = predicted_velocity



    def calculate_data(self, sprint, sprint_data, time_delay=0):
        """Helper function to be used when iterating through each sprint -  adds new parameters to each sprint_dict"""
        sprint['acceleration'] = sprint_data['MSS'] / sprint_data['tau'] * np.e**((-sprint['time']- time_delay)/sprint_data['tau'])
        sprint['position'] = sprint_data['MSS'] * (sprint['time']  + sprint_data['tau'] * np.e**((-sprint['time']-time_delay)/sprint_data['tau'])) - sprint_data['MSS'] * sprint_data['tau']
        sprint['Fhzt'] = self.bw * sprint['acceleration']
        sprint['Fair'] = 0.33 * sprint['predicted_velocity']**2
        sprint['Ftot'] = sprint['Fhzt'] + sprint['Fair']
        sprint['Ftot_kg'] = sprint['Ftot'] / self.bw
        sprint['PowerHzt_kg'] = sprint['Ftot_kg'] * sprint['predicted_velocity']
        sprint['Fres'] = np.sqrt(sprint['Fhzt']**2 + (self.bw * 9.81)**2)
        sprint['RF_perc'] = np.where(sprint['time'] < 0.3, 0, sprint['Fhzt']/sprint['Fres'])
        sprint_data['name'] = self.name
        sprint_data['bodyweight'] = self.bw

        return sprint, sprint_data



    def apply_calculations(self):
        """Pass each sprint and its associated parameters to the caluclate_data function"""
        for i, (sprint, sprint_data) in enumerate(zip(self.sprints, self.sprint_dict.keys())):
            self.sprints[i], self.sprint_dict[sprint_data] = self.calculate_data(sprint, self.sprint_dict[sprint_data])




    def model_parameters(self, sprint, sprint_data):
        """Helper function to calculate further sprint parameters including: F0, V0, Pmax"""
        F0_model = LinearRegression()
        F0_model.fit(sprint['predicted_velocity'].values.reshape(-1,1), sprint['Ftot'].values.reshape(-1,1))
        sprint_data['F0'] = F0_model.intercept_[0]
        F0_kg_model = LinearRegression()
        F0_kg_model.fit(sprint['predicted_velocity'].values.reshape(-1,1), sprint['Ftot_kg'].values.reshape(-1,1))
        sprint_data['F0_kg'] = sprint_data['F0'] / self.bw
        sprint_data['FV_profile'] = F0_kg_model.coef_[0][0]
        sprint_data['V0'] = -sprint_data['F0_kg'] / sprint_data['FV_profile']
        sprint_data['Pmax'] = sprint_data['F0'] * sprint_data['V0'] / 4
        sprint_data['Pmax_kg'] = (sprint_data['Pmax'] / self.bw)
        sprint_data['RF_max'] = np.max(sprint['RF_perc'])
        DRF_kg_model = LinearRegression()
        DRF_kg_model.fit(sprint['predicted_velocity'].values.reshape(-1,1), sprint['RF_perc'].values.reshape(-1,1))
        sprint_data['DRF'] = DRF_kg_model.coef_[0][0]

        return sprint_data



    def calculate_params(self):
        """"Pass each sprint and its associated parameters to the caluclate_parameters function"""
        for i, (sprint, sprint_data) in enumerate(zip(self.sprints, self.sprint_dict.keys())):
            self.sprint_dict[sprint_data] = self.model_parameters(sprint, self.sprint_dict[sprint_data])


    def plot_single_plot(self, sprint, sprint_data, plot_title=None, plot_name=None):
        """helper funtion to plot a series of subplots graphing the sprint data"""

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24,12))
        fig.suptitle(f"{plot_title} - {self.name}", fontsize="xx-large")

        axcolor1 = "tab:red"
        axcolor2 = "tab:blue"
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_title("Force & Speed Profile")
        #axes[0,0].set_ylim(0,5) # NEED TO SET BASED OFF PROPER DATA
        axes[0,0].scatter(sprint['time'], sprint['vel'], c='black')
        axes[0,0].plot(sprint['time'], sprint['predicted_velocity'], c='red')
        axes[0,0].set_xlabel('Time (s)', )
        axes[0,0].set_ylabel('Running Speed (m/s)', color=axcolor1)
        axes0 = axes[0,0].twinx()
        #axes0.grid(True)
        #axes0.set_ylim(0,5)
        axes0.plot(sprint['time'], sprint['acceleration'], c='blue')
        axes0.set_ylabel('Acceleration (m/s)', color=axcolor2)


        axcolor1 = "tab:red"
        axcolor2 = "tab:gray"
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_title("Force & Power Profile")
        axes[0,1].plot(sprint['time'], sprint['predicted_velocity'], c='red')
        axes[0,1].plot(sprint['time'], sprint['Ftot_kg'], c='gray')
        axes[0,1].set_xlabel("Time (s)")
        axes[0,1].set_ylabel("Force (N/kg) and Speed (m/s)", color=axcolor1)
        axes1 = axes[0,1].twinx()
        axes1.plot(sprint['time'], sprint['PowerHzt_kg'], c='black')
        axes1.set_ylabel("Power (W/kg)", color=axcolor2)

        axcolor1 = "tab:gray"
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_title("Field Sprint Force-Velocity-Power Profile")
        axes[1,0].plot(sprint['predicted_velocity'], sprint['Ftot_kg'], c='gray')
        axes[1,0].set_xlabel("Velocity (m/s)")
        axes[1,0].set_ylabel("Force (N/kg)", color=axcolor1)
        axes2 = axes[1,0].twinx()
        axes2.plot(sprint['predicted_velocity'], sprint['PowerHzt_kg'], c='black')
        axes2.set_ylabel("Power (W/kg)", color=axcolor2)

        axcolor1 = "tab:blue"
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_title("Horizontal Force Velocity Profiles")
        axes[1,1].plot(sprint['predicted_velocity'][sprint['time'] > 0.3], sprint['RF_perc'][sprint['time'] > 0.3], c='blue')
        axes[1,1].set_xlabel("Velocity (m/s)")
        axes[1,1].set_ylabel("Ratio Force (%)", color=axcolor1)

        #plt.savefig(plot_name)


    def plot_multiple_plots(self):
        """Plot a series of subplots for each file and for each sprint"""
        for i, (sprint, sprint_data) in enumerate(zip(self.sprints, self.sprint_dict.keys())):
            file_name = f"{dt.now().strftime('%Y%m%d')}_{self.name}_{i}"
            self.plot_single_plot(sprint, self.sprint_dict[sprint_data], plot_title=sprint_data, plot_name=file_name)
            self.sprint_dict[sprint_data]["plot_name"] = file_name

    def create_report(self, template="template.html"):
        """Create a report using Jinja template"""
        df_list = [{key: value} for key, value in self.sprint_dict.items()]
        df = pd.DataFrame.from_dict(self.sprint_dict).T.drop("plot_name", axis=1)
        df = df.astype('float').round(2)
        plot_names = [self.sprint_dict[sprint]['plot_name'] for sprint in self.sprint_dict.keys()]

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
        template = env.get_template(template)
        html = template.render(data=df.style.set_precision(2).render(),
                              #data=df.to_html(),
                              player_name=self.name,
                              plots=plot_names)

        # Write the HTML file
        with open(f'{dt.now().strftime("%Y%m%d")}_{self.name}.html', 'w') as f:
            f.write(html)


