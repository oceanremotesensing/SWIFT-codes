# import statements
import numpy as np
import matplotlib.pyplot as plt
from seaSurfaceDynamicsEstimator import seaSurfaceDynamicsEstimator_linear

class seaSurfaceDynamics:
    def __init__(self, x, y, z, time, dt):
        # Assign position of buoy at each point in space
        self.time = time
        self.dt = dt
        self.x = x
        self.y = y
        self.z = z

        # Compute Velocity arrays from positions
        self.u = np.gradient(self.x) # cross-shore velocity, units are meters/sec
        self.v = np.gradient(self.y) # alongshore velocity, units are meters/sec
        self.w = np.gradient(self.z) # vertical velocity, units are meters/sec

        # Compute Acceleration arrays from velocities
        self.ax = np.gradient(self.u) # cross-shore acceleration, units are meters/sec^2
        self.ay = np.gradient(self.v) # Along shore acceleration, units are meters/sec^2
        self.az = np.gradient(self.w) # vertical acceleration, units are meters/sec^2

        # Define the overall state
        self.true_state = np.array([self.x, self.y, self.z, self.u, self.v, self.w, self.ax, self.ay, self.az])

    def microSWIFTMeasurments(self):
        ## ASSUMPTIONS:
        # 1. All measurements are gaussian distributed with no bias 
        # 2. All measurements are independent events

        # Define microSWIFT instrument characteristics
        gps_pos_std = .1         # lat, lon accuracy standard deviation, units are meters
        gps_altitude_std = .1    # GPS alitude accuracy standard deviation, units are meters
        gps_vel_std = .1         # GPS velocity standard deviation, units are meters/sec
        imu_accel_std = .04      # IMU acceleration standard deviation, units are meters/sec^2
        imu_gyro_str = .1        # IMU gyroscope standard deviation, units are degrees/sec

        # GPS Measurements
        # Compute measurments from random samples in a normal distribution with each instruments standard deviation
        self.x_meas = np.random.normal(loc=0, scale=gps_pos_std, size=np.shape(self.x)) + self.x
        self.y_meas = np.random.normal(loc=0, scale=gps_pos_std, size=np.shape(self.y)) + self.y
        self.z_meas = np.random.normal(loc=0, scale=gps_altitude_std, size=np.shape(self.z)) + self.z

        # Compute Velocity measurements
        self.u_meas = np.random.normal(loc=0, scale=gps_vel_std, size=np.shape(self.u)) + self.u
        self.v_meas = np.random.normal(loc=0, scale=gps_vel_std, size=np.shape(self.v)) + self.v
        self.w_meas = np.gradient(self.z_meas)

        # IMU Measurements
        self.ax_meas = np.random.normal(loc=0, scale=imu_accel_std, size=np.shape(self.ax)) + self.ax
        self.ay_meas = np.random.normal(loc=0, scale=imu_accel_std, size=np.shape(self.ay)) + self.ay
        self.az_meas = np.random.normal(loc=0, scale=imu_accel_std, size=np.shape(self.az)) + self.az

        # Define measured state
        self.measured_state = np.array([self.x_meas, self.y_meas, self.z_meas, self.u_meas, self.v_meas, self.w_meas, self.ax_meas, self.ay_meas, self.az_meas])

    def plotTrueState(self):
        fig_simple_signals, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
        # x and y position
        ax1.plot(self.time, self.x)
        ax1.plot(self.time, self.y)
        ax1.set_ylabel('Position [meters]')

        # Sea Surface Elevation signals
        ax2.plot(self.time, self.z)
        ax2.set_ylabel('Sea Surface Elevation [meters]')

        # Velocity signals
        ax3.plot(self.time, self.u)
        ax3.plot(self.time, self.v)
        ax3.plot(self.time, self.w)
        ax3.set_ylabel('Velocity [m/s]')

        # Acceleration Signals
        ax4.plot(self.time, self.ax)
        ax4.plot(self.time, self.ay)
        ax4.plot(self.time, self.az)
        ax4.set_ylabel('Acceleration [m/s^2]')
        ax4.set_xlabel('Time [seconds]')

        # Show the plot
        plt.show()

    def plotMeasuredState(self):
        # Plot the measured values
        fig_simple_measured, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
        # x and y position
        ax1.plot(self.time, self.x_meas)
        ax1.plot(self.time, self.y_meas)
        ax1.set_ylabel('Position Measured [meters]')

        # Sea Surface Elevation signals
        ax2.plot(self.time, self.z_meas)
        ax2.set_ylabel('Sea Surface Elevation Measured [meters]')

        # Velocity signals
        ax3.plot(self.time, self.u_meas)
        ax3.plot(self.time, self.v_meas)
        ax3.set_ylabel('Velocity Measured [m/s]')

        # Acceleration Signals
        ax4.plot(self.time, self.ax_meas)
        ax4.plot(self.time, self.ay_meas)
        ax4.plot(self.time, self.az_meas)
        ax4.set_ylabel('Acceleration [m/s^2]')
        ax4.set_xlabel('Time [seconds]')
        plt.show()

    def compareEstimate2True(self):
        fig_est, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
        # x and y position
        ax1.plot(self.time, self.state_est[0,:])
        ax1.plot(self.time, self.state_est[1,:])
        ax1.set_ylabel('Position [meters]')

        # Sea Surface Elevation signals
        ax2.plot(self.time, self.state_est[2,:])
        ax2.set_ylabel('Sea Surface Elevation [meters]')
                
        # Velocity signals
        ax3.plot(self.time, self.state_est[3,:])
        ax3.plot(self.time, self.state_est[4,:])
        ax3.set_ylabel('Velocity Measured [m/s]')

        # Acceleration Signals
        ax4.plot(self.time, self.state_est[6,:])
        ax4.plot(self.time, self.state_est[7,:])
        ax4.plot(self.time, self.state_est[8,:])
        ax4.set_ylabel('Acceleration [m/s^2]')
        ax4.set_xlabel('Time [seconds]')
        plt.show()

        # Compute percent error and plot
        error = (self.state_est - self.true_state)/self.true_state * 100
        fig_error, ax = plt.subplots()
        ax.plot(error[0,:])
        plt.show()
        

def main():
    '''
    @edwinrainville

    Description: This script is run to test the sea surface dynamics estimator algorithms that are developed.

    '''

    # Define time array for all tests
    sampling_freq = 12 # Sampling frequency of the dataset, units are Hz
    dt = 1/sampling_freq # time between samples, units are seconds
    seconds = 1000
    time = np.linspace(0,seconds, num=seconds*sampling_freq) # Time in seconds

    ## Test 1 (Simple)- simple cosine function for surface buoys moved in a straight line on surface
    wave_period = 10 # Period of the test wave, units are seconds
    wave_amp = 2    # Amplitude of wave, units are meters
    x_simple_true = wave_amp * np.cos(((2*np.pi + np.pi/2)/wave_period) * time) + np.arange(200, 500, step=(500-200)/len(time)) # straight line, units are meters
    y_simple_true = wave_amp * np.cos(((2*np.pi + np.pi/2)/wave_period) * time) + np.arange(100, 200, step=(200-100)/len(time)) # straight line, units are meters
    z_simple_true = wave_amp * np.cos(((2*np.pi)/wave_period) * time) # cosine function, units are meters

    # Instantiate simple sea class (singular cosine function)
    simple_sea = seaSurfaceDynamics(x_simple_true, y_simple_true, z_simple_true, time, dt)
    simple_sea.microSWIFTMeasurments()
    simple_sea.plotTrueState()
    simple_sea.plotMeasuredState()

    # seaSurfaceDynamicsEstimator test
    simple_sea.state_est = seaSurfaceDynamicsEstimator_linear(simple_sea.measured_state, simple_sea.dt)
    simple_sea.compareEstimate2True()

    ## Test 2 (Two_components)- two simple cosine functions for surface and buoys moved in a straight line on surface
    wave_period_1 = 10 # Period of the test wave, units are seconds
    wave_amp_1 = 2    # Amplitude of wave, units are meters
    wave_period_2 = 15
    wave_amp_2 = 1
    x_two_comp_true = wave_amp_1 * np.cos(((2*np.pi + np.pi/2)/wave_period_1) * time) + wave_amp_2 * np.cos(((2*np.pi + np.pi/2)/wave_period_2) * time) + np.arange(200, 500, step=(500-200)/len(time)) # straight line, units are meters
    y_two_comp_true = wave_amp_1 * np.cos(((2*np.pi + np.pi/2)/wave_period_1) * time) + wave_amp_2 * np.cos(((2*np.pi + np.pi/2)/wave_period_2) * time) + np.arange(100, 200, step=(200-100)/len(time)) # straight line, units are meters
    z_two_comp_true = wave_amp_1 * np.cos(((2*np.pi)/wave_period_1) * time) + wave_amp_2 * np.cos(((2*np.pi + np.pi/2)/wave_period_2) * time) # cosine function, units are meters

    # Instantiate simple sea class (singular cosine function)
    two_comp_sea = seaSurfaceDynamics(x_two_comp_true, y_two_comp_true, z_two_comp_true, time, dt)
    two_comp_sea.microSWIFTMeasurments()
    two_comp_sea.plotTrueState()
    two_comp_sea.plotMeasuredState()

    # seaSurfaceDynamicsEstimator test
    two_comp_sea.state_est = seaSurfaceDynamicsEstimator_linear(two_comp_sea.measured_state, two_comp_sea.dt)
    two_comp_sea.compareEstimate2True()

    ## Test 3 (Three components)- three simple cosine functions for surface and buoys moved in a straight line on surface
    wave_period_1 = 10 # Period of the test wave, units are seconds
    wave_amp_1 = 2    # Amplitude of wave, units are meters
    wave_period_2 = 15
    wave_amp_2 = 1
    wave_period_3 = 3
    wave_amp_3 = 0.5

    # Wave Components
    comp_1 = wave_amp_1 * np.cos(((2*np.pi + np.pi/2)/wave_period_1) * time)
    comp_2 = wave_amp_2 * np.cos(((2*np.pi + np.pi/2)/wave_period_2) * time)
    comp_3 = wave_amp_3 * np.cos(((2*np.pi + np.pi/2)/wave_period_3) * time)
    
    # Position time series
    x_three_comp_true =  comp_1 + comp_2 + comp_3 + np.arange(200, 500, step=(500-200)/len(time)) 
    y_three_comp_true =  comp_1 + comp_2 + comp_3 + np.arange(100, 200, step=(200-100)/len(time))
    z_three_comp_true =  comp_1 + comp_2 + comp_3 

    # Instantiate simple sea class (singular cosine function)
    three_comp_sea = seaSurfaceDynamics(x_three_comp_true, y_three_comp_true, z_three_comp_true, time, dt)
    three_comp_sea.microSWIFTMeasurments()
    three_comp_sea.plotTrueState()
    three_comp_sea.plotMeasuredState()

    # seaSurfaceDynamicsEstimator test
    three_comp_sea.state_est = seaSurfaceDynamicsEstimator_linear(three_comp_sea.measured_state, three_comp_sea.dt)
    three_comp_sea.compareEstimate2True()

if __name__=='__main__':
    main()