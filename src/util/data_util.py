from dataclasses import dataclass
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, freqs, filtfilt, freqz
from pathlib import Path
from math import pi, pow

# DATA_DIR = "src/data_processing/data/"
DATA_DIR = "data/"


@dataclass
class ThreeAxisData:
    x: np.ndarray = 0
    y: np.ndarray = 0
    z: np.ndarray = 0


@dataclass
class EulerData:
    roll: np.ndarray = 0
    pitch: np.ndarray = 0
    yaw: np.ndarray = 0


@dataclass
class QuatData:
    w: np.ndarray = 0
    x: np.ndarray = 0
    y: np.ndarray = 0
    z: np.ndarray = 0


@dataclass
class IMUData:
    accel_data: ThreeAxisData = ThreeAxisData()
    gyro_data: ThreeAxisData = ThreeAxisData()
    mag_data: ThreeAxisData = ThreeAxisData()


@dataclass
class AHRSData:
    euler: EulerData = EulerData()
    quat: QuatData = EulerData()


class Sensor:
    timestamp: np.ndarray = 0
    _DEFAULT_SINGLE_FIG_SIZE = (10, 3)
    _DEFAULT_DOUBLE_FIG_SIZE = (10, 6)
    _DEFAULT_TRIPLE_FIG_SIZE = (10, 9)
    _DEFAULT_PRESSURE_YLABEL = "Pressure [kPa]"
    _DEFAULT_SURFACE_YLABEL = "Surface [m]"
    _DEFAULT_ACCEL_MSS_YLABEL = "Acceleration [m/s/s]"
    _DEFAULT_ACCEL_G_YLABEL = "Acceleration [g]"
    _DEFAULT_GYRO_YLABEL = "Rotation Rate [deg/s]"
    _DEFAULT_MAG_YLABEL = "Magnetic Field [mG]"

    fs: int
    fn: int 

    def __init__(self, 
                sensor_id: str,
                sensor_name: str,
                data_filename: str,
                fs: int) -> None:
        self.sensor_id = sensor_id
        self.sensor_name = sensor_name
        self.data_filename = data_filename
        self.data_path = DATA_DIR + data_filename
        self.fs = fs
        self.fn = fs/2

    def import_data(self, start_index:int=None, end_index:int=None):            
        pass


    def filter_time_series(self, signal: np.ndarray, **kwargs):
        _N = kwargs.get("N", 1)
        _fc = kwargs.get("fc", self.fn)
        _Wn = _fc / self.fn
        _btype = kwargs.get("btype", "low")
        _analog = kwargs.get("analog", False)
        _output = kwargs.get("output", "ba")
        
        if _output == "ba":
            b, a = butter(_N, _Wn, _btype, _analog, _output, fs=self.fs)
            w, h = freqz(b, a)
            self.plot_freq_response(w, h, fc=_fc)

            return filtfilt(b, a, signal)



    def plot_time_series(self, signal:np.ndarray, ax: plt.Axes = None, **kwargs):
        _figsize = kwargs.get("figsize", self._DEFAULT_SINGLE_FIG_SIZE)
        _title = kwargs.get("title", '')
        _xlab = kwargs.get("xlabel", '')
        _ylab = kwargs.get("ylabel", '')
        _xlim = kwargs.get("xlim", (None, None))
        _ylim = kwargs.get("ylim", (None, None))
        
        _f = None
        _ax = ax
        if not ax:
            _f, _ax = plt.subplots(figsize=_figsize)
        _ax.set_title(_title)
        _ax.plot(self.timestamp, signal)
        _ax.set_xlabel(_xlab)
        _ax.set_ylabel(_ylab)
        _ax.set_xlim(_xlim)
        _ax.set_ylim(_ylim)
        return (_f, _ax)

    
    def plot_ndim_time_series(self, signal:tuple, ax: plt.Axes = None, **kwargs):
        _ax = ax
        for sig in signal:
            _fig, _ax = self.plot_time_series(sig, _ax, **kwargs)
        return _fig, _ax

    
    def plot_freq_response(self, w, h, **kwargs):
        _fig_size = kwargs.get("figsize", self._DEFAULT_DOUBLE_FIG_SIZE)
        _fc = kwargs.get("fc", self.fn)

        _fig, (_ax_amp, _ax_phase) = plt.subplots(2, 1, figsize = _fig_size, sharex=True)
        _fig.suptitle("Butterwroth Filter Frequency Response")
        _fig.supxlabel("Frequency [radians / second]")

        _ax_amp.semilogx(w, 20 * np.log10(abs(h)))
        _ax_amp.set_ylabel('Amplitude [dB]')
        _ax_amp.margins(0, 0.1)
        _ax_amp.grid(which='both', axis='both')
        _ax_amp.axvline(_fc, color='green') # cutoff frequency

        _angles = np.unwrap(np.angle(h))
        _ax_phase.semilogx(w, _angles)
        _ax_phase.set_ylabel('Angle (radians)')
        _ax_phase.grid(which='both', axis='both')
        _ax_phase.axvline(_fc, color='green') # cutoff frequency


class HOBOSensor(Sensor):
    pressure: np.ndarray = 0
    temperature: np.ndarray = 0
    surface: np.ndarray = 0
    mwl: float = 0

    upcross_times: np.ndarray = 0
    waves: list = []

    def __init__(self, sensor_id: str, sensor_name: str, data_filename: str, fs: int=1, rho: int=1025, g: float=9.81) -> None:
        super().__init__(sensor_id, sensor_name, data_filename, fs)
        self._rho = rho
        self._g = g


    def import_data(self, start_index: int=None, end_index: int=None, **kwargs):
        # Input handling
        _trim_start = kwargs.get("trim_start", 10)
        _trim_end = kwargs.get("trim_end", 10)
        _pressure_threshold = kwargs.get("threshold", 110)
        
        # Import time series to instance memory
        df = pd.read_csv(self.data_path, names=["Timestamp", "Abs Pressure", "Temperature"], header=2, usecols=[1,2,3], skiprows=start_index, nrows=end_index)
        self.timestamp = pd.to_datetime(df["Timestamp"]).to_numpy() # Convert the timestamps to a numpy vector and save to the instance
        self.temperature = df["Temperature"].to_numpy()             # Save the temperature values to the instance
        _abs_pressure = df["Abs Pressure"].to_numpy()               # Read in the absolute pressure values

        # Determine if pressure data is in imperial or metric
        if np.mean(_abs_pressure < 100):                        # If pressure data is imperial, then the mean should be really low
            _abs_pressure *= 6.894757                           # Convert from PSI to kPa
        if np.mean(self.temperature > 40):                      # If temperature data is imperial, then the mean should be really high
            self.temperature = (self.temperature - 32) * (9/5)  # Convert from °F to °C

        # Parse data and save
        _water_values = _abs_pressure[_abs_pressure>_pressure_threshold]                                # Determine where the sensor is in the water
        _water_indices = np.nonzero(np.in1d(_abs_pressure, _water_values))[0][_trim_start:-_trim_end]   # Determine the indices of the water pressure values and subtract 10 values from each end
        _atmo_pressure = np.mean(_abs_pressure[1:190])                                                  # Determine the atmospheric pressure from the initial line of data
        self.timestamp = self.timestamp[_water_indices]                                                 # Trim the pressure data to an acceptable range
        self.pressure = _water_values[_trim_start:-_trim_end] - _atmo_pressure                          # Convert to gauge pressure
        (self.surface, self.mwl) = self.pressure_to_eta(self.pressure)                                  # Convert the pressure data to surface displacement and detrend it; also save the MWL

    
    def plot_all_data(self, **kwargs):
        _figsize = kwargs.get("figsize", self._DEFAULT_DOUBLE_FIG_SIZE)
 
        _fig, (_ax_press, _ax_eta) = plt.subplots(2, 1, constrained_layout=True, sharex=True, figsize=_figsize)
        _fig.suptitle(self.sensor_name)

        self.plot_pressure_series(_ax_press, **kwargs)
        self.plot_surface_series(_ax_eta, **kwargs)


    def plot_pressure_series(self, ax: plt.Axes=None, **kwargs):
        return super().plot_time_series(self.pressure, ax, title="Gauge Pressure Over Time", ylabel=self._DEFAULT_PRESSURE_YLABEL, **kwargs)

    
    def plot_surface_series(self, ax: plt.Axes=None, **kwargs):
        return super().plot_time_series(self.surface, ax, title="Surface Elevation Over Time", ylabel=self._DEFAULT_SURFACE_YLABEL, **kwargs)

    
    def pressure_to_eta(self, pressure: np.ndarray):
        mwl = np.mean(pressure)*1000 / (self._rho * self._g)
        eta = np.divide(detrend(pressure)*1000, (self._rho * self._g))
        return (eta, mwl)
    

    def wave_by_wave(self):
        _wave = {"start time": None, 
                 "surface": None,
                 "height": None, 
                 "period": None,
                 "length": None,
                 "iribarren": None}
        
        _index = 0 # Initialize upcross indexes
        neg_data_locs = np.where(self.surface < 0)[0] # Generate index array of negative values
        for i in range(0, len(neg_data_locs)): # Iterate through all negative values to find upcrossings
            try:
                if self.surface[neg_data_locs[i]+1] >= 0: # Identify locations of upcrossings
                    self.upcross_times = np.append(self.upcross_times, neg_data_locs[i])
                    _index += 1
            except:
                continue
        
        for i in range(1,len(self.upcross_times)-1):
            _wave["start time"] = self.upcross_times[i]
            _wave["surface"] = self.surface[int(self.upcross_times[i]):int(self.upcross_times[i+1])]
            _wave["height"] = np.max(_wave["surface"]) - np.min(_wave["surface"])
            _wave["period"] = (self.upcross_times[i+1] - self.upcross_times[i]) / self.fs
            _wave["length"] = self._g / (2*pi) * pow(_wave["period"], 2)
            self.waves.append(dict(_wave))

        # Calculate height data
        _heights = [w["height"] for w in self.waves]
        H_avg = np.mean(_heights)
        _heights = np.sort(_heights)[::-1] # Sort the wave heights in descending order
        H_sig = np.mean(_heights[1:round(len(_heights)/2)])
        H_rms = np.sqrt(np.mean(np.square(_heights)))

        # Calculate period data
        _periods = [w["period"] for w in self.waves]
        T_avg = np.mean(_periods)
        _periods = np.sort(_periods)[::-1] # Sort the wave heights in descending order
        T_sig = np.mean(_periods[1:round(len(_periods)/2)])
        T_rms = np.sqrt(np.mean(np.square(_periods)))

        return (H_avg, H_sig, H_rms, T_avg, T_sig, T_rms)
    

    def plot_wave_by_wave(self, **kwargs):
        fig, ax = plt.subplots(1,1, figsize=self._DEFAULT_SINGLE_FIG_SIZE)
        _, ax = super().plot_time_series(self.surface, ax, title="Surface Elevation with Upcrossing Markers", ylabel=self._DEFAULT_SURFACE_YLABEL, **kwargs)
        ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyles='--', colors='k')
        ax.vlines(self.timestamp[self.upcross_times], ax.get_ylim()[0], ax.get_ylim()[1], linestyles='--', colors='r')

        return (fig, ax)
    

    def plot_histogram(self, **kwargs):
        _nbins = kwargs.get("nbins", 20)

        _heights = [w["height"] for w in self.waves]

        fig, ax = plt.subplots(1, 1, figsize=self._DEFAULT_SINGLE_FIG_SIZE)
        fig.suptitle("Wave Height Histogram " + self.sensor_name)
        ax.hist(_heights, _nbins, edgecolor='k')
        ax.set_xlabel("Wave Height [m]")
        ax.set_ylabel("Num. of Occurrences")

        return fig, ax
    

    def plot_pdfcdf(self, **kwargs):
        _nbins = kwargs.get("nbins", 20)
        _ylim = kwargs.get("ylim", (None, None))

        _heights = [w["height"] for w in self.waves]

        x: np.ndarray = np.array([])
        counts, edges = np.histogram(_heights, bins=_nbins)
        for i in range(0, len(edges)-1): # For every bin edge value
            x = np.append(x, (edges[i+1] + edges[i]) / 2) # Determine the bin center from the average of the edges
        pdf = counts / len(_heights)
        cdf = np.cumsum(pdf)

        fig, ax = plt.subplots(1, 1, figsize=self._DEFAULT_SINGLE_FIG_SIZE)
        fig.suptitle("Probability/Cumulative Distribution Function")
        ax.plot(x, pdf)
        ax.plot(x, cdf)
        ax.set_xlabel("Wave Height [m]")
        ax.set_ylabel("Probability of Occurrence")
        ax.legend(["PDF", "CDF"])
        ax.set_ylim(_ylim[0], _ylim[1])
        
        ax2 = ax.twinx()

        return fig, ax, pdf, cdf





class LowellSensor(Sensor):
    imu_data = IMUData()
    ahrs_data = AHRSData()

    def __init__(self, sensor_id: str, sensor_name: str, data_filename: str, fs: int=16) -> None:
        super().__init__(sensor_id, sensor_name, data_filename, fs)


    def import_data(self, start_index: int = None, end_index: int = None):
        df = pd.read_csv(self.data_path, names=["ISO 8601 Time", "Ax (g)", "Ay (g)", "Az (g)", "Mx (mG)", "My (mG)", "Mz (mG)"], header=1, skiprows=start_index, nrows=end_index)
        self.timestamp = pd.to_datetime(df["ISO 8601 Time"]).to_numpy()
        _accel = ThreeAxisData(df["Ax (g)"].to_numpy(), df["Ay (g)"].to_numpy(), df["Az (g)"].to_numpy())
        _mag = ThreeAxisData(df["Mx (mG)"].to_numpy(), df["My (mG)"].to_numpy(), df["Mz (mG)"].to_numpy())
        self.imu_data = IMUData(_accel, ThreeAxisData(), _mag)


    def plot_all_data(self, **kwargs):
        _figsize = kwargs.get("figsize", self._DEFAULT_DOUBLE_FIG_SIZE)
 
        _fig, (_ax_accel, _ax_mag) = plt.subplots(2, 1, constrained_layout=True, sharex=True, figsize=_figsize)
        _fig.suptitle(self.sensor_name)

        _, _ax_accel = self.plot_accel_series(_ax_accel, **kwargs)
        _, _ax_mag = self.plot_mag_series(_ax_mag, **kwargs)

        _ax_accel.legend(["X-axis", "Y-axis", "Z-axis"])
        _ax_mag.legend(["X-axis", "Y-axis", "Z-axis"])
        
        return _fig, (_ax_accel, _ax_mag)

    
    def plot_accel_series(self, ax: plt.Axes=None, **kwargs):
        return super().plot_ndim_time_series((self.imu_data.accel_data.x, self.imu_data.accel_data.y, self.imu_data.accel_data.z), ax, title="Accelerometer Data Over Time", ylabel=self._DEFAULT_ACCEL_G_YLABEL, **kwargs)


    def plot_mag_series(self, ax: plt.Axes=None, **kwargs):
        return super().plot_ndim_time_series((self.imu_data.mag_data.x, self.imu_data.mag_data.y, self.imu_data.mag_data.z), ax, title="Magnetometer Data Over Time", ylabel=self._DEFAULT_MAG_YLABEL, **kwargs)


class ThetisSensor(Sensor):
    imu_data = IMUData()
    ahrs_data = AHRSData()

    def __init__(self, sensor_id: str, sensor_name: str, data_filename: str, fs: int=16) -> None:
        super().__init__(sensor_id, sensor_name, data_filename, fs)


    def import_data(self, start_index: int = None, end_index: int = None):
        pass


    def import_gps_data(self, filename: str, start_index: int = None, end_index: int = None):
        pass

    
    def import_imu_data(self, filename: str, start_index: int = None, end_index: int = None):
        df = pd.read_csv(DATA_DIR + filename, names=["Timestamp", "sysCal", "gyroCal", "accelCal", "magCal", "rawAccelX", "rawAccelY", "rawAccelZ", "accelX", "accelY", "accelZ","rawGyroX", "rawGyroY", "rawGyroZ", "gyroX", "gyroY", "gyroZ", "rawMagX", "rawMagY","rawMagZ", "magX", "magY", "magZ", "linAccelX", "linAccelY", "linAccelZ"], header=1, skiprows=start_index, nrows=end_index)
        self.timestamp = pd.to_datetime(df["Timestamp"]).to_numpy()
        _accel = ThreeAxisData(df["rawAccelX"].to_numpy(), df["rawAccelY"].to_numpy(), df["rawAccelZ"].to_numpy())
        _gyro = ThreeAxisData(df["rawGyroX"].to_numpy(), df["rawGyroY"].to_numpy(), df["rawGyroZ"].to_numpy())
        _mag = ThreeAxisData(df["rawMagX"].to_numpy(), df["rawMagY"].to_numpy(), df["rawMagZ"].to_numpy())
        self.imu_data = IMUData(_accel, _gyro, _mag)

    
    def import_ahrs_data(self, filename: str, start_index: int = None, end_index: int = None):
        df = pd.read_csv(DATA_DIR + filename, names=["Timestamp", "roll", "pitch", "yaw", "quatW", "quatX", "quatY", "quatZ"], header=1, skiprows=start_index, nrows=end_index)
        self.timestamp_data = pd.to_datetime(df["Timestamp"]).to_numpy()
        _euler = EulerData(df["roll"].to_numpy(), df["pitch"].to_numpy(), df["yaw"].to_numpy())
        _quat = QuatData(df["quatW"].to_numpy(), df["quatX"].to_numpy(), df["quatY"].to_numpy(), df["quatZ"].to_numpy())
        self.ahrs_data = AHRSData(_euler, _quat)


    def plot_imu_data(self, **kwargs):
        _figsize = kwargs.get("figsize", self._DEFAULT_TRIPLE_FIG_SIZE)
 
        _fig, (_ax_accel, _ax_gyro, _ax_mag) = plt.subplots(3, 1, constrained_layout=True, sharex=True, figsize=_figsize)
        _fig.suptitle(self.sensor_name)

        _, _ax_accel = self.plot_accel_series(_ax_accel, **kwargs)
        _, _ax_gyro = self.plot_gyro_series(_ax_gyro, **kwargs)
        _, _ax_mag = self.plot_mag_series(_ax_mag, **kwargs)

        _ax_accel.legend(["X-axis", "Y-axis", "Z-axis"])
        _ax_gyro.legend(["X-axis", "Y-axis", "Z-axis"])
        _ax_mag.legend(["X-axis", "Y-axis", "Z-axis"])
        
        return _fig, (_ax_accel, _ax_mag)


    def plot_accel_series(self, ax: plt.Axes=None, **kwargs):
        return super().plot_ndim_time_series((self.imu_data.accel_data.x, self.imu_data.accel_data.y, self.imu_data.accel_data.z), ax, title="Accelerometer Data Over Time", ylabel=self._DEFAULT_ACCEL_G_YLABEL, **kwargs)


    def plot_gyro_series(self, ax: plt.Axes=None, **kwargs):
        return super().plot_ndim_time_series((self.imu_data.gyro_data.x, self.imu_data.gyro_data.y, self.imu_data.gyro_data.z), ax, title="Gyroscope Data Over Time", ylabel=self._DEFAULT_GYRO_YLABEL, **kwargs)


    def plot_mag_series(self, ax: plt.Axes=None, **kwargs):
        return super().plot_ndim_time_series((self.imu_data.mag_data.x, self.imu_data.mag_data.y, self.imu_data.mag_data.z), ax, title="Magnetometer Data Over Time", ylabel=self._DEFAULT_MAG_YLABEL, **kwargs)