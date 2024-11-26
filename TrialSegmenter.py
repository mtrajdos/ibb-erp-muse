import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib.gridspec import GridSpec
import tkinter as tk 
from tkinter import filedialog, messagebox
import logging
import traceback
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from scipy.stats import sem

class MuseTrialProcessor:
    def __init__(self):
        self.GUI = {
            'LAYOUT': {
                'figure_size': (15, 12),  # Increased height for additional subplot
                'left_margin': 0.15,
                'right_margin': 0.85,
                'top_margin': 0.95,
                'bottom_margin': 0.1,
                'subplot_spacing': 0.3
            },
            'CONTROLS': {
                'filter_toggles': {
                    'x': 0.88,
                    'y': 0.8,
                    'width': 0.1,
                    'height': 0.1
                },
                'scale_slider': {
                    'x': 0.88,
                    'y': 0.6,
                    'width': 0.1,
                    'height': 0.02
                },
                'prev_button': {
                    'x': 0.88,
                    'y': 0.4,
                    'width': 0.1,
                    'height': 0.04
                },
                'next_button': {
                    'x': 0.88,
                    'y': 0.3,
                    'width': 0.1,
                    'height': 0.04
                },
                'avg_button': {
                    'x': 0.88,
                    'y': 0.2,
                    'width': 0.1,
                    'height': 0.04
                }
            }
        }
        
        self.CONFIG = {
            'PHOTODIODE': {
                'DETECTION': {
                    'THRESHOLD': 300,
                    'MIN_AMPLITUDE': 50,
                    'SLOPE_DIRECTION': 'pos',
                    'MIN_DURATION': 0.02,
                    'MAX_DURATION': 0.1,
                    'SEARCH_WINDOW': 0.5,
                    'CHANNEL': 'AUX_R',
                },
                'VALIDATION': {
                    'MIN_SAMPLES': 1,
                    'MAX_OFFSET': 500,
                }
            },
            'SAMPLING_RATE': 256,
            'TRIAL_WINDOW': {
                'START': -0.5,
                'END': 0.5,
            },
            'CHANNELS': {
                'EEG': ['AF7', 'AF8', 'TP9', 'TP10'],
                'AUX': ['AUX_L', 'AUX_R']
            },
            'FILTERS': {
                'EEG': {
                    'enabled': True,
                    'highpass': 1,
                    'lowpass': 15,
                    'notch': 50,
                    'method': 'fir',
                    'design': 'firwin',
                    'window': 'hamming',
                    'phase': 'zero',
                    'order': '5s',
                    'pad': 'reflect_limited'
                },
                'AUX': {
                    'enabled': True,
                    'highpass': None,
                    'lowpass': None,
                    'notch': None,
                    'method': 'fir',
                    'design': 'firwin',
                    'window': 'hamming',
                    'phase': 'zero',
                    'order': 'auto',
                    'pad': 'reflect_limited'
                }
            },
            'BASELINE': {
                'ON': True,
                'WINDOW_START': -0.2,
                'WINDOW_END': 0.0
            },
            'ERP': {
                'EPN': {
                    'window': [150, 300],
                    'channels': ['TP9', 'TP10'],
                    'smoothing': {
                        'enabled': True,
                        'method': 'savgol',
                        'window': 6,
                        'order': 2
                    }
                },
                'LPP': {
                    'window': [400, 800],
                    'channels': ['AF7', 'AF8'],
                    'smoothing': {
                        'enabled': True,
                        'method': 'savgol',
                        'window': 4,
                        'order': 2
                    }
                }
            },
            'PLOT': {
                'SCALES': {
                    'EEG': {
                        'default': 20,
                        'max': 1700,
                        'min': 1
                    },
                    'AUX': {
                        'default': 1000,
                        'max': 1800,
                        'min': -100
                    }
                },
                'COLORS': {
                    'AF7': '#4169E1',
                    'AF8': '#4169E1',
                    'TP9': '#32CD32',
                    'TP10': '#32CD32',
                    'AUX_L': '#9467bd',
                    'AUX_R': '#8c564b',
                    'checkerboard195': '#1f77b4',
                    'checkerboard225': '#2ca02c',
                    'checkerboard255': '#ff7f0e'
                },
                'STYLES': {
                    'background': '#F5F5F7',
                    'grid_alpha': 0.15,
                    'line_width': 2,
                    'marker_alpha': 0.5,
                    'axis_label_pad': 10,
                    'title_pad': 15
                }
            },
            'VISUALIZATION': {
                'STIM_WINDOW_ALPHA': 0.2,
                'STIM_WINDOW_COLOR': 'gray'
            },
            'AVERAGING': {
                'conditions': ['checkerboard195', 'checkerboard225', 'checkerboard255', 'all']
            },
            'LOGGING': {
                'file_level': logging.DEBUG,
                'console_level': logging.INFO,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

        # Setup logging
        self._setup_logging()
        
        # Initialize attributes
        self._initialize_attributes()

    def _setup_logging(self):
        """Setup detailed logging configuration."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=f'trialSegmenter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
        self.logger = logging.getLogger('muse_processor')
        mne.set_log_level('WARNING')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

    def _initialize_attributes(self):
        """Initialize all class attributes."""
        self.CHANNELS = self.CONFIG['CHANNELS']
        self.CHANNEL_TYPES = {ch: 'EEG' for ch in self.CHANNELS['EEG']}
        self.CHANNEL_TYPES.update({ch: 'AUX' for ch in self.CHANNELS['AUX']})
        
        # Data containers
        self.raw_data = None
        self.filtered_data = None
        self.trials = None
        self.timestamps = None
        self.experiment_start_time = None
        self.input_filepath = None
        self.log_data = None
        self.SF = self.CONFIG['SAMPLING_RATE']
        self.avg_stim_duration = None

        # GUI state
        self.current_trial = 0
        self.current_time = 0.0
        self.current_condition = 0
        self.epoch_length = self.CONFIG['TRIAL_WINDOW']['END'] - self.CONFIG['TRIAL_WINDOW']['START']
        
        self.channel_scales = {
            ch: self.CONFIG['PLOT']['SCALES']['EEG']['default'] 
            for ch in self.CHANNELS['EEG']
        }
        self.channel_scales.update({
            ch: self.CONFIG['PLOT']['SCALES']['AUX']['default'] 
            for ch in self.CHANNELS['AUX']
        })

        # Initialize electrode selection state
        self.electrode_states = {
            'AF7': True,
            'AF8': True,
            'TP9': False,
            'TP10': False
        }

        # Scales for different plot components
        self.y_scale_erp = self.CONFIG['PLOT']['SCALES']['EEG']['default']
        self.y_scale_pd = self.CONFIG['PLOT']['SCALES']['AUX']['default']

    def load_muse_data(self, filepath):
        """Load and preprocess MUSE EEG data."""
        self.logger.info(f"Loading MUSE data from {filepath}")
        self.input_filepath = filepath
        
        try:
            # Read raw data
            data = {
                'timestamp': [],
                'TP9': [], 'AF7': [], 'AF8': [], 'TP10': [],
                'AUX_L': [], 'AUX_R': []
            }
            
            total_lines = sum(1 for _ in open(filepath))
            
            with open(filepath, 'r') as f:
                for line in tqdm(f, total=total_lines, desc="Loading MUSE data"):
                    try:
                        parts = line.strip().split(',')
                        if '/muse/eeg' in parts[1]:
                            timestamp = float(parts[0])
                            if not data['timestamp']:
                                self.experiment_start_time = timestamp
                            
                            data['timestamp'].append(timestamp)
                            data['TP9'].append(float(parts[2]))
                            data['AF7'].append(float(parts[3]))
                            data['AF8'].append(float(parts[4]))
                            data['TP10'].append(float(parts[5]))
                            data['AUX_L'].append(float(parts[6]))
                            data['AUX_R'].append(float(parts[7]))
                    except Exception as e:
                        self.logger.warning(f"Skipped line due to error: {e}")
                        continue
            
            # Convert to DataFrame and interpolate missing values
            df = pd.DataFrame(data)
            self.timestamps = df['timestamp'].values
            
            # Create MNE raw object
            ch_types = ['eeg'] * len(self.CHANNELS['EEG']) + ['misc'] * len(self.CHANNELS['AUX'])
            ch_names = self.CHANNELS['EEG'] + self.CHANNELS['AUX']
            
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=self.CONFIG['SAMPLING_RATE'],
                ch_types=ch_types
            )
            
            data_array = df[ch_names].to_numpy().T
            self.raw_data = mne.io.RawArray(data_array, info)
            
            self.logger.info(f"Data loaded successfully: {len(self.timestamps)} samples")
            
        except Exception as e:
            self.logger.error(f"Error loading MUSE data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def load_log_file(self, filepath):
        """Load experimental log file with updated format."""
        self.logger.info(f"Loading log file from {filepath}")
        try:
            # Read the CSV file into a DataFrame
            self.log_data = pd.read_csv(filepath)
            
            # Calculate average stimulus duration
            self.avg_stim_duration = self.log_data['Stim_Duration'].mean()
            self.logger.info(f"Average stimulus duration: {self.avg_stim_duration:.3f}s")
            
            # Extract stimulus types
            self.log_data['StimulusType'] = self.log_data['StimFile'].apply(
                lambda x: x.split('.')[0]
            )
            
            self.logger.info(f"Log file loaded successfully: {len(self.log_data)} entries")
            
            # Validate data
            if not all(col in self.log_data.columns for col in 
                    ['StimON', 'StimOFF', 'Stim_Duration']):
                raise ValueError("Log file missing required columns")
            
            # Add validation for timing consistency
            timing_checks = {
                'duration_match': np.allclose(
                    self.log_data['StimOFF'] - self.log_data['StimON'],
                    self.log_data['Stim_Duration'],
                    rtol=1e-3
                ),
                'duration_range': all(
                    (0.5 < d < 1.0) for d in self.log_data['Stim_Duration']
                )
            }
            
            for check_name, result in timing_checks.items():
                if not result:
                    self.logger.warning(f"Timing validation failed: {check_name}")
                
        except Exception as e:
            self.logger.error(f"Error loading log file: {e}")
            raise

    def detect_photodiode_events(self, data, approx_time=None):
        """Detect photodiode events with focus on rising edges."""
        ch_names = data.ch_names
        aux_idx = ch_names.index(self.CONFIG['PHOTODIODE']['DETECTION']['CHANNEL'])
        aux_data = data.get_data()[aux_idx]
        
        # Calculate gradients
        gradient = np.gradient(aux_data)
        smooth_gradient = np.convolve(gradient, np.ones(5)/5, mode='same')
        
        if approx_time is not None:
            # Search window for specific time point
            center_sample = int(approx_time * self.SF)
            window_size = int(0.5 * self.SF)
            window_start = max(0, center_sample - window_size)
            window_end = min(len(aux_data), center_sample + window_size)
            
            # Extract window data
            aux_data = aux_data[window_start:window_end]
            smooth_gradient = smooth_gradient[window_start:window_end]
            
        # Find potential triggers (rising edges only)
        rising_mask = smooth_gradient > self.CONFIG['PHOTODIODE']['DETECTION']['MIN_AMPLITUDE']
        rising_edges = np.where(rising_mask)[0]
        
        if len(rising_edges) == 0:
            return None if approx_time is not None else []
        
        # Group consecutive edges
        groups = []
        current_group = [rising_edges[0]]
        
        for idx in rising_edges[1:]:
            if idx - current_group[-1] <= 3:
                current_group.append(idx)
            else:
                if len(current_group) >= self.CONFIG['PHOTODIODE']['VALIDATION']['MIN_SAMPLES']:
                    groups.append(current_group)
                current_group = [idx]
                
        if len(current_group) >= self.CONFIG['PHOTODIODE']['VALIDATION']['MIN_SAMPLES']:
            groups.append(current_group)
        
        # Validate triggers
        valid_triggers = []
        for group in groups:
            # Find peak gradient
            group_gradients = smooth_gradient[group]
            peak_idx = group[np.argmax(group_gradients)]
            
            # Calculate signal levels
            pre_window = slice(max(0, peak_idx-10), peak_idx)
            post_window = slice(peak_idx, min(len(aux_data), peak_idx+10))
            
            pre_trigger = np.mean(aux_data[pre_window])
            post_trigger = np.mean(aux_data[post_window])
            
            # Comprehensive validation
            if all([
                (post_trigger - pre_trigger) > self.CONFIG['PHOTODIODE']['DETECTION']['MIN_AMPLITUDE'],
                np.max(smooth_gradient[group]) > abs(np.min(smooth_gradient[post_window])),
                np.mean(aux_data[post_window]) > np.mean(aux_data[pre_window])
            ]):
                trigger_idx = peak_idx if approx_time is None else window_start + peak_idx
                valid_triggers.append(trigger_idx)
        
        if approx_time is not None:
            if valid_triggers:
                # Select best trigger based on timing
                trigger_times = np.array(valid_triggers) / self.SF
                time_diffs = np.abs(trigger_times - approx_time)
                return valid_triggers[np.argmin(time_diffs)]
            return None
            
        return np.array(valid_triggers)

    def create_trials(self):
        """Create trials from detected photodiode events."""
        self.logger.info("Creating trials with photodiode alignment...")
        
        if not hasattr(self, 'raw_data') or not hasattr(self, 'log_data'):
            raise ValueError("Raw data and log data must be loaded first")
        
        events = []
        event_id = {}
        timing_stats = []
        
        for idx, row in self.log_data.iterrows():
            scene_time = float(row['StimON'])
            relative_time = scene_time - self.experiment_start_time
            
            event_sample = self.detect_photodiode_events(
                self.filtered_data or self.raw_data, 
                approx_time=relative_time
            )
            
            if event_sample is not None:
                # Create event code based on stimulus type
                stim_type = row['StimulusType']
                if stim_type not in event_id:
                    event_id[stim_type] = len(event_id) + 1
                
                events.append([event_sample, 0, event_id[stim_type]])
                
                # Calculate timing information
                event_time = event_sample / self.SF
                offset_ms = (event_time - relative_time) * 1000
                timing_stats.append(offset_ms)
                
                self.logger.debug(
                    f"Trial {idx+1}: Event at {scene_time:.3f}s "
                    f"(offset: {offset_ms:.1f}ms)"
                )
            else:
                self.logger.warning(
                    f"Trial {idx+1}: No valid event found at {scene_time:.3f}s"
                )
        
        if not events:
            raise ValueError("No valid photodiode-aligned events found")
            
        events = np.array(events)
        
        # Store photodiode timing statistics
        self.photodiode_stats = {
            'stats': {
                'mean_offset': np.mean(timing_stats),
                'std_offset': np.std(timing_stats),
                'min_offset': np.min(timing_stats),
                'max_offset': np.max(timing_stats),
                'count': len(events),
                'all_offsets': timing_stats
            }
        }
        
        # Create epochs
        self.trials = mne.Epochs(
            self.filtered_data or self.raw_data,
            events,
            event_id=event_id,
            tmin=self.CONFIG['TRIAL_WINDOW']['START'],
            tmax=self.CONFIG['TRIAL_WINDOW']['END'],
            baseline=(self.CONFIG['BASELINE']['WINDOW_START'], 
                     self.CONFIG['BASELINE']['WINDOW_END']) if self.CONFIG['BASELINE']['ON'] else None,
            preload=True,
            reject_by_annotation=True,
            verbose=False
        )
        
        self.logger.info(f"""
        Created {len(self.trials)} epochs with timing statistics:
        - Mean offset: {self.photodiode_stats['stats']['mean_offset']:.1f}ms
        - Std offset: {self.photodiode_stats['stats']['std_offset']:.1f}ms
        - Range: {self.photodiode_stats['stats']['min_offset']:.1f}ms to {self.photodiode_stats['stats']['max_offset']:.1f}ms
        """)

    def apply_filters(self):
        """Apply configured filters to the data."""
        try:
            self.filtered_data = self.raw_data.copy()
            
            if self.CONFIG['FILTERS']['EEG']['enabled']:
                self.filtered_data.filter(
                    l_freq=self.CONFIG['FILTERS']['EEG']['highpass'],
                    h_freq=self.CONFIG['FILTERS']['EEG']['lowpass'],
                    picks=self.CHANNELS['EEG'],
                    method=self.CONFIG['FILTERS']['EEG']['method'],
                    phase=self.CONFIG['FILTERS']['EEG']['phase']
                )
            
            if self.CONFIG['FILTERS']['AUX']['enabled']:
                if self.CONFIG['FILTERS']['AUX']['highpass'] or self.CONFIG['FILTERS']['AUX']['lowpass']:
                    self.filtered_data.filter(
                        l_freq=self.CONFIG['FILTERS']['AUX']['highpass'],
                        h_freq=self.CONFIG['FILTERS']['AUX']['lowpass'],
                        picks=self.CHANNELS['AUX'],
                        method=self.CONFIG['FILTERS']['AUX']['method'],
                        phase=self.CONFIG['FILTERS']['AUX']['phase']
                    )
            
        except Exception as e:
            self.logger.error(f"Error in filter application: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def setup_gui(self):
        """Setup the visualization GUI."""
        self.fig = plt.figure(figsize=self.GUI['LAYOUT']['figure_size'])
        self.fig.patch.set_facecolor(self.CONFIG['PLOT']['STYLES']['background'])
        
        gs = GridSpec(len(self.CHANNELS['EEG'] + self.CHANNELS['AUX']), 1)
        self.axes = {}
        
        for i, ch in enumerate(self.CHANNELS['EEG'] + self.CHANNELS['AUX']):
            ax = self.fig.add_subplot(gs[i, 0])
            ax.set_facecolor('white')
            ax.grid(True, alpha=self.CONFIG['PLOT']['STYLES']['grid_alpha'])
            
            self.axes[ch] = ax
            ax.set_ylabel(
                f'{ch}\n(μV)',
                color=self.CONFIG['PLOT']['COLORS'][ch],
                labelpad=self.CONFIG['PLOT']['STYLES']['axis_label_pad']
            )
            
            if i == len(self.CHANNELS['EEG'] + self.CHANNELS['AUX']) - 1:
                ax.set_xlabel('Time (ms)')
        
        self.setup_controls()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.update_plot()

    def setup_controls(self):
            """Setup GUI controls."""
            ctrl_cfg = self.GUI['CONTROLS']
            
            # Filter toggles
            filter_pos = ctrl_cfg['filter_toggles']
            filter_ax = plt.axes([
                filter_pos['x'], filter_pos['y'],
                filter_pos['width'], filter_pos['height']
            ])
            self.filter_checks = CheckButtons(
                filter_ax,
                ['EEG Filters', 'AUX Filters'],
                [self.CONFIG['FILTERS']['EEG']['enabled'],
                self.CONFIG['FILTERS']['AUX']['enabled']]
            )
            
            # Scale slider
            slider_pos = ctrl_cfg['scale_slider']
            scale_ax = plt.axes([
                slider_pos['x'], slider_pos['y'],
                slider_pos['width'], slider_pos['height']
            ])
            self.scale_slider = Slider(
                scale_ax, 'Scale (μV)',
                self.CONFIG['PLOT']['SCALES']['EEG']['min'],
                self.CONFIG['PLOT']['SCALES']['EEG']['max'],
                valinit=self.CONFIG['PLOT']['SCALES']['EEG']['default']
            )
            
            # Navigation buttons
            self.prev_button = Button(
                plt.axes([ctrl_cfg['prev_button']['x'], ctrl_cfg['prev_button']['y'],
                        ctrl_cfg['prev_button']['width'], ctrl_cfg['prev_button']['height']]),
                '◀ Previous'
            )
            self.next_button = Button(
                plt.axes([ctrl_cfg['next_button']['x'], ctrl_cfg['next_button']['y'],
                        ctrl_cfg['next_button']['width'], ctrl_cfg['next_button']['height']]),
                'Next ▶'
            )
            self.avg_button = Button(
                plt.axes([ctrl_cfg['avg_button']['x'], ctrl_cfg['avg_button']['y'],
                        ctrl_cfg['avg_button']['width'], ctrl_cfg['avg_button']['height']]),
                'Show Averages'
            )
            
            # Connect callbacks
            self.filter_checks.on_clicked(self.toggle_filters)
            self.scale_slider.on_changed(self.update_scale)
            self.prev_button.on_clicked(lambda x: self.navigate_trials('prev'))
            self.next_button.on_clicked(lambda x: self.navigate_trials('next'))
            # Fix the callback connection for averages button
            self.avg_button.on_clicked(lambda x: self.setup_averages_view())

    def update_plot(self):
            """Update the plot with current trial data."""
            if self.trials is None or len(self.trials) == 0:
                self.logger.warning("No trials available to plot")
                for ax in self.axes.values():
                    ax.clear()
                    ax.text(0.5, 0.5, 'No valid trials available',
                        ha='center', va='center')
                plt.draw()
                return
                
            try:
                data = self.trials.get_data()[self.current_trial]
                times = self.trials.times * 1000  # Convert to milliseconds
                
                # Get trial information
                event_code = self.trials.events[self.current_trial, 2]
                event_type = list(self.trials.event_id.keys())[
                    list(self.trials.event_id.values()).index(event_code)
                ]
                
                # Update title
                trial_num = self.current_trial + 1
                stim_file = self.log_data.iloc[self.current_trial]['StimFile']
                title = f"Trial {trial_num}/{len(self.trials)} - {stim_file}"
                
                if hasattr(self, 'photodiode_stats'):
                    stats = self.photodiode_stats['stats']
                    title += f"\nTiming - Mean: {stats['mean_offset']:.1f}ms, "
                    title += f"Std: {stats['std_offset']:.1f}ms"
                self.fig.suptitle(title, y=0.98)
                
                # Plot stimulus window
                stim_start = 0
                stim_end = self.avg_stim_duration * 1000  # Convert to ms
                
                # Update each channel plot
                for ch in self.CHANNELS['EEG'] + self.CHANNELS['AUX']:
                    ax = self.axes[ch]
                    ax.clear()
                    
                    ch_idx = self.trials.ch_names.index(ch)
                    ch_type = 'EEG' if ch in self.CHANNELS['EEG'] else 'AUX'
                    
                    # Plot stimulus window
                    ax.axvspan(stim_start, stim_end,
                            alpha=self.CONFIG['VISUALIZATION']['STIM_WINDOW_ALPHA'],
                            color=self.CONFIG['VISUALIZATION']['STIM_WINDOW_COLOR'],
                            zorder=1)
                    
                    # Plot data
                    ax.plot(times, data[ch_idx],
                        color=self.CONFIG['PLOT']['COLORS'][ch],
                        linewidth=self.CONFIG['PLOT']['STYLES']['line_width'],
                        zorder=2)
                    
                    # Add vertical line at trigger
                    ax.axvline(x=0, color='r', linestyle='--',
                            alpha=self.CONFIG['PLOT']['STYLES']['marker_alpha'])
                    
                    # Set axis limits
                    ax.set_xlim(
                        self.CONFIG['TRIAL_WINDOW']['START'] * 1000,
                        self.CONFIG['TRIAL_WINDOW']['END'] * 1000
                    )
                    
                    # Set y-axis limits based on channel type
                    if ch_type == 'EEG':
                        ax.set_ylim(-self.channel_scales[ch], self.channel_scales[ch])
                    else:  # AUX
                        scale_range = self.channel_scales[ch] / 2
                        midpoint = (self.CONFIG['PLOT']['SCALES']['AUX']['max'] +
                                self.CONFIG['PLOT']['SCALES']['AUX']['min']) / 2
                        ax.set_ylim(midpoint - scale_range, midpoint + scale_range)
                    
                    # Style
                    ax.grid(True, alpha=self.CONFIG['PLOT']['STYLES']['grid_alpha'])
                    ax.set_ylabel(
                        f'{ch}\n(μV)',
                        color=self.CONFIG['PLOT']['COLORS'][ch],
                        labelpad=self.CONFIG['PLOT']['STYLES']['axis_label_pad']
                    )
                    
                    if ch == list(self.axes.keys())[-1]:
                        ax.set_xlabel('Time (ms)')
                        
                plt.draw()
                
            except Exception as e:
                self.logger.error(f"Error updating plot: {str(e)}")
                self.logger.error(traceback.format_exc())

    def setup_averages_view(self, event=None):
            """Setup interactive averages plot."""
            if self.trials is None:
                self.logger.warning("No trials available for averaging")
                return
                
            # Create new figure
            self.fig_avg = plt.figure(figsize=self.GUI['LAYOUT']['figure_size'])
            gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
            
            # Create subplots
            self.ax_avg = self.fig_avg.add_subplot(gs[0])  # ERP plot
            self.ax_pd = self.fig_avg.add_subplot(gs[1])   # Photodiode plot
            
            # Add condition selection to include 'Grand Average'
            self.CONFIG['AVERAGING']['conditions'] = [
                'checkerboard195', 
                'checkerboard225', 
                'checkerboard255', 
                'Grand Average'
            ]
            
            # Initialize electrode state
            if not hasattr(self, 'electrode_states'):
                self.electrode_states = {
                    'AF7': True,
                    'AF8': True,
                    'TP9': False,
                    'TP10': False
                }
            
            # Setup control panel
            ctrl_width = 0.2
            main_width = 0.98 - ctrl_width
            
            # Navigation controls
            btn_height = 0.04
            btn_spacing = 0.02
            current_y = 0.85
            
            # Previous/Next buttons
            self.prev_button_avg = Button(
                plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
                '◀ Previous'
            )
            current_y -= (btn_height + btn_spacing)
            
            self.next_button_avg = Button(
                plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
                'Next ▶'
            )
            current_y -= (btn_height + btn_spacing*2)

            # Add condition label
            self.condition_label = self.fig_avg.text(
                main_width + 0.01 + (ctrl_width-0.02)/2,
                current_y,
                'Current View',
                horizontalalignment='center'
            )
            current_y -= btn_spacing
            
            # Electrode selection
            self.fig_avg.text(
                main_width + 0.01 + (ctrl_width-0.02)/2,
                current_y,
                'Select Electrodes',
                horizontalalignment='center'
            )
            current_y -= btn_spacing
            
            check_height = 0.15
            self.electrode_checks = CheckButtons(
                plt.axes([main_width + 0.01, current_y - check_height, ctrl_width-0.02, check_height]),
                ['AF7/AF8', 'TP9/TP10'],
                [True, False]
            )
            current_y -= (check_height + btn_spacing*2)
            
            # Scale controls
            self.zoom_in_erp = Button(
                plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
                'ERP Zoom +'
            )
            current_y -= (btn_height + btn_spacing)
            
            self.zoom_out_erp = Button(
                plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
                'ERP Zoom -'
            )
            current_y -= (btn_height + btn_spacing*2)
            
            self.zoom_in_pd = Button(
                plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
                'PD Zoom +'
            )
            current_y -= (btn_height + btn_spacing)
            
            self.zoom_out_pd = Button(
                plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
                'PD Zoom -'
            )
            
            # Connect callbacks
            self.prev_button_avg.on_clicked(lambda x: self.navigate_conditions('prev'))
            self.next_button_avg.on_clicked(lambda x: self.navigate_conditions('next'))
            self.zoom_in_erp.on_clicked(self.zoom_in_erp_plot)
            self.zoom_out_erp.on_clicked(self.zoom_out_erp_plot)
            self.zoom_in_pd.on_clicked(self.zoom_in_pd_plot)
            self.zoom_out_pd.on_clicked(self.zoom_out_pd_plot)
            self.electrode_checks.on_clicked(self.toggle_electrodes)
            
            # Style figure
            self.fig_avg.patch.set_facecolor(self.CONFIG['PLOT']['STYLES']['background'])
            self.ax_avg.set_facecolor('white')
            self.ax_pd.set_facecolor('white')
            
            # Adjust layout
            self.fig_avg.subplots_adjust(right=main_width)
            
            # Initialize and update
            self.current_condition = 0
            self.update_averages_plot()
            plt.show()

    def update_averages_plot(self):
        """Update the averages plot."""
        self.ax_avg.clear()
        self.ax_pd.clear()
        
        conditions = self.CONFIG['AVERAGING']['conditions']
        condition = conditions[self.current_condition]
        times = self.trials.times * 1000  # Convert to milliseconds
        
        # Add stimulus window to all plots
        stim_window = [0, self.avg_stim_duration * 1000]  # Convert to ms
        for ax in [self.ax_avg, self.ax_pd]:
            ax.axvspan(
                stim_window[0], stim_window[1],
                alpha=self.CONFIG['VISUALIZATION']['STIM_WINDOW_ALPHA'],
                color=self.CONFIG['VISUALIZATION']['STIM_WINDOW_COLOR'],
                zorder=1
            )
        
        # Plot condition-specific averages or grand average
        if condition == 'Grand Average':
            all_data = []
            n_trials = 0
            # Collect data from all conditions
            for cond in [c for c in conditions if c != 'Grand Average']:
                cond_epochs = self.trials[cond]
                for pair in [('AF7', 'AF8'), ('TP9', 'TP10')]:
                    if not any(self.electrode_states[ch] for ch in pair):
                        continue
                    for ch in pair:
                        ch_idx = self.trials.ch_names.index(ch)
                        all_data.append(cond_epochs.get_data()[:, ch_idx, :])
                n_trials += len(cond_epochs)
            
            # Calculate grand average
            if all_data:
                all_data = np.concatenate(all_data, axis=0)
                grand_avg = np.mean(all_data, axis=0)
                grand_stderr = sem(all_data, axis=0)
                
                # Plot grand average
                self.ax_avg.plot(times, grand_avg, color='#2ca02c', 
                               label='Grand Average',
                               linewidth=self.CONFIG['PLOT']['STYLES']['line_width'])
                self.ax_avg.fill_between(
                    times, grand_avg - grand_stderr, grand_avg + grand_stderr,
                    color='#2ca02c', alpha=0.2
                )
        else:
            cond_epochs = self.trials[condition]
            self._plot_condition_average(cond_epochs, times, condition)
        
        # Plot photodiode average
        pd_idx = self.trials.ch_names.index(self.CONFIG['PHOTODIODE']['DETECTION']['CHANNEL'])
        pd_data = self.trials.get_data()[:, pd_idx, :]
        pd_avg = np.mean(pd_data, axis=0)
        pd_sem = sem(pd_data, axis=0)
        
        self.ax_pd.plot(
            times, pd_avg,
            color=self.CONFIG['PLOT']['COLORS'][self.CONFIG['PHOTODIODE']['DETECTION']['CHANNEL']],
            linewidth=self.CONFIG['PLOT']['STYLES']['line_width'],
            label='Photodiode'
        )
        self.ax_pd.fill_between(
            times, pd_avg - pd_sem, pd_avg + pd_sem,
            color=self.CONFIG['PLOT']['COLORS'][self.CONFIG['PHOTODIODE']['DETECTION']['CHANNEL']],
            alpha=0.2
        )
        
        # Style plots
        self._style_average_plots()
        
        # Update condition label
        if hasattr(self, 'condition_label'):
            self.condition_label.set_text(f'Current View: {condition}')
        
        self.fig_avg.canvas.draw_idle()

    def _plot_condition_average(self, epochs, times, condition):
            """Plot average for a specific condition."""
            for pair in [('AF7', 'AF8'), ('TP9', 'TP10')]:
                if not any(self.electrode_states[ch] for ch in pair):
                    continue
                    
                data = []
                for ch in pair:
                    ch_idx = epochs.ch_names.index(ch)
                    ch_data = epochs.get_data()[:, ch_idx, :]
                    data.append(ch_data)
                
                # Average across channels first, then trials
                all_data = np.mean(data, axis=0)
                avg = np.mean(all_data, axis=0)
                stderr = sem(all_data, axis=0)
                
                color = self.CONFIG['PLOT']['COLORS'][condition] if condition != 'Grand Average' else '#2ca02c'
                label = f'{pair[0]}/{pair[1]} ({condition})'
                
                self.ax_avg.plot(times, avg, color=color, label=label,
                            linewidth=self.CONFIG['PLOT']['STYLES']['line_width'])
                self.ax_avg.fill_between(
                    times, avg - stderr, avg + stderr,
                    color=color, alpha=0.2
                )

    def _plot_grand_average(self, times):
        """Plot grand average across all conditions."""
        all_conditions = [c for c in self.CONFIG['AVERAGING']['conditions'] if c != 'all']
        
        for pair in [('AF7', 'AF8'), ('TP9', 'TP10')]:
            if not any(self.electrode_states[ch] for ch in pair):
                continue
                
            all_data = []
            for condition in all_conditions:
                cond_epochs = self.trials[condition]
                pair_data = []
                for ch in pair:
                    ch_idx = cond_epochs.ch_names.index(ch)
                    ch_data = cond_epochs.get_data()[:, ch_idx, :]
                    pair_data.append(ch_data)
                # Average across channels
                all_data.append(np.mean(pair_data, axis=0))
            
            # Stack all condition data
            all_data = np.vstack(all_data)
            
            # Calculate grand average and SEM
            grand_avg = np.mean(all_data, axis=0)
            grand_stderr = sem(all_data, axis=0)
            
            color = self.CONFIG['PLOT']['COLORS'][pair[0]]
            label = f'{pair[0]}/{pair[1]} (Grand Average)'
            
            self.ax_all.plot(times, grand_avg, color=color, label=label,
                           linewidth=self.CONFIG['PLOT']['STYLES']['line_width'])
            self.ax_all.fill_between(
                times, grand_avg - grand_stderr, grand_avg + grand_stderr,
                color=color, alpha=0.2
            )

    def _style_average_plots(self):
            """Apply styling to average plots."""
            # Style common elements for both plots
            for ax in [self.ax_avg, self.ax_pd]:
                # Add trigger line
                ax.axvline(x=0, color='black', linestyle='--',
                        alpha=self.CONFIG['PLOT']['STYLES']['marker_alpha'])
                
                # Add horizontal zero line
                ax.axhline(y=0, color='black', linestyle='-',
                        alpha=0.3, zorder=1)
                
                # Set common x limits
                ax.set_xlim(
                    self.CONFIG['TRIAL_WINDOW']['START'] * 1000,
                    self.CONFIG['TRIAL_WINDOW']['END'] * 1000
                )
                
                # Add grid
                ax.grid(True, alpha=self.CONFIG['PLOT']['STYLES']['grid_alpha'])
            
            # Set specific y limits and labels
            self.ax_avg.set_ylim(-self.y_scale_erp, self.y_scale_erp)
            self.ax_pd.set_ylim(-self.y_scale_pd, self.y_scale_pd)
            
            # Set titles and labels
            condition = self.CONFIG['AVERAGING']['conditions'][self.current_condition]
            n_trials = len(self.trials) if condition == 'Grand Average' else len(self.trials[condition])
            self.ax_avg.set_title(f'{condition} (n={n_trials})')
            
            self.ax_avg.set_ylabel('ERP Amplitude (μV)')
            self.ax_pd.set_ylabel('Photodiode (μV)')
            self.ax_pd.set_xlabel('Time (ms)')
            
            # Add legends if any electrodes selected
            if any(self.electrode_states.values()):
                self.ax_avg.legend(loc='upper right')
            self.ax_pd.legend(loc='upper right')
        
    def navigate_conditions(self, direction):
        """Navigate between conditions in average view."""
        conditions = self.CONFIG['AVERAGING']['conditions']
        if direction == 'prev':
            self.current_condition = (self.current_condition - 1) % len(conditions)
        elif direction == 'next':
            self.current_condition = (self.current_condition + 1) % len(conditions)
        
        self.update_averages_plot()

    def navigate_trials(self, direction):
        """Navigate between trials."""
        if direction == 'prev' and self.current_trial > 0:
            self.current_trial -= 1
        elif direction == 'next' and self.current_trial < len(self.trials) - 1:
            self.current_trial += 1
            
        self.update_plot()

    def toggle_electrodes(self, label):
        """Handle electrode pair toggling."""
        if label == 'AF7/AF8':
            self.electrode_states['AF7'] = not self.electrode_states['AF7']
            self.electrode_states['AF8'] = self.electrode_states['AF7']
        else:  # TP9/TP10
            self.electrode_states['TP9'] = not self.electrode_states['TP9']
            self.electrode_states['TP10'] = self.electrode_states['TP9']
        
        self.update_averages_plot()

    def toggle_filters(self, label):
        """Toggle filters on/off."""
        if label == 'EEG Filters':
            self.CONFIG['FILTERS']['EEG']['enabled'] = not self.CONFIG['FILTERS']['EEG']['enabled']
        else:
            self.CONFIG['FILTERS']['AUX']['enabled'] = not self.CONFIG['FILTERS']['AUX']['enabled']
            
        self.apply_filters()
        if self.trials is not None:
            self.create_trials()
        self.update_plot()

    def update_scale(self, value):
        """Update plotting scale."""
        self.scale = value
        self.update_plot()

    # Zoom control methods
    def zoom_in_erp_plot(self, event):
        self.y_scale_erp = max(
            self.CONFIG['PLOT']['SCALES']['EEG']['min'],
            self.y_scale_erp * 0.8
        )
        self.update_averages_plot()

    def zoom_out_erp_plot(self, event):
        self.y_scale_erp = min(
            self.CONFIG['PLOT']['SCALES']['EEG']['max'],
            self.y_scale_erp * 1.2
        )
        self.update_averages_plot()

    def zoom_in_pd_plot(self, event):
        self.y_scale_pd = max(
            self.CONFIG['PLOT']['SCALES']['AUX']['min'],
            self.y_scale_pd * 0.8
        )
        self.update_averages_plot()

    def zoom_out_pd_plot(self, event):
        self.y_scale_pd = min(
            self.CONFIG['PLOT']['SCALES']['AUX']['max'],
            self.y_scale_pd * 1.2
        )
        self.update_averages_plot()

    # Event handlers
    def on_scroll(self, event):
        """Handle scroll events."""
        if event.inaxes:
            ch = event.inaxes.get_ylabel().split('\n')[0]
            ch_type = 'EEG' if ch in self.CHANNELS['EEG'] else 'AUX'
            scales = self.CONFIG['PLOT']['SCALES'][ch_type]
            
            current_scale = self.channel_scales[ch]
            
            if event.button == 'up':
                new_scale = current_scale / 1.2
            else:
                new_scale = current_scale * 1.2
            
            if ch_type == 'EEG':
                self.channel_scales[ch] = np.clip(new_scale, scales['min'], scales['max'])
            else:  # AUX
                min_scale = (scales['max'] - scales['min']) / 10
                max_scale = scales['max'] - scales['min']
                self.channel_scales[ch] = np.clip(new_scale, min_scale, max_scale)
            
            self.update_plot()

    def on_scroll_avg(self, event):
        """Handle scroll events in average view."""
        if event.inaxes:
            if event.inaxes == self.ax_avg:
                if event.button == 'up':
                    self.zoom_in_erp_plot(None)
                else:
                    self.zoom_out_erp_plot(None)
            elif event.inaxes == self.ax_pd:
                if event.button == 'up':
                    self.zoom_in_pd_plot(None)
                else:
                    self.zoom_out_pd_plot(None)

    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'left':
            self.navigate_trials('prev')
        elif event.key == 'right':
            self.navigate_trials('next')
        elif event.key == 'up':
            self.scale_slider.set_val(min(
                self.CONFIG['PLOT']['SCALES']['EEG']['max'],
                self.scale * 1.2
            ))
        elif event.key == 'down':
            self.scale_slider.set_val(max(
                self.CONFIG['PLOT']['SCALES']['EEG']['min'],
                self.scale / 1.2
            ))

    def on_key_press_avg(self, event):
        """Handle keyboard events in average view."""
        if event.key == 'left':
            self.navigate_conditions('prev')
        elif event.key == 'right':
            self.navigate_conditions('next')
        elif event.key == 'up':
            if event.inaxes == self.ax_avg:
                self.zoom_in_erp_plot(None)
            elif event.inaxes == self.ax_pd:
                self.zoom_in_pd_plot(None)
        elif event.key == 'down':
            if event.inaxes == self.ax_avg:
                self.zoom_out_erp_plot(None)
            elif event.inaxes == self.ax_pd:
                self.zoom_out_pd_plot(None)

def main():
    """Main entry point for the application."""
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Create processor instance
        processor = MuseTrialProcessor()
        
        # Get input files
        muse_file = filedialog.askopenfilename(
            title="Select MUSE data file",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not muse_file:
            print("No MUSE file selected. Exiting...")
            return
            
        log_file = filedialog.askopenfilename(
            title="Select log file",
            filetypes=[
                ("All Log Files", ("*.txt", "*.log", "*.csv")),
                ("Text files", "*.txt"),
                ("Log files", "*.log"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if not log_file:
            print("No log file selected. Exiting...")
            return
            
        # Process data
        processor.load_muse_data(muse_file)
        processor.load_log_file(log_file)
        processor.apply_filters()
        processor.create_trials()
        processor.setup_gui()
        
        plt.show()
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        messagebox.showerror("Error", str(e))
        raise

if __name__ == "__main__":
    main()
