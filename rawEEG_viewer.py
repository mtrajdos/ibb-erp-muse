import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='muse_viewer_debug.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class MuseViewer:
    def __init__(self, filepath: str):
        self.logger = logging.getLogger(__name__)
        
        # Constants
        self.SF = 256
        self.DEFAULT_EPOCH = 5.0
        self.DEFAULT_SCALE = 150
        self.CHANNELS = {
            'eeg': ['TP9', 'AF7', 'AF8', 'TP10'],
            'aux': ['Right AUX', 'Left AUX']
        }
        self.COLORS = {
            'TP9': '#1f77b4',      # Blue
            'AF7': '#ff7f0e',      # Orange
            'AF8': '#2ca02c',      # Green
            'TP10': '#d62728',     # Red
            'Right AUX': '#9467bd', # Purple
            'Left AUX': '#8c564b'   # Brown
        }
        
        self.filepath = filepath
        self._load_data()
        
        # Center the data initially
        self.center_data()
        
        # Store both raw and filtered data
        self.original_data = self.raw.get_data().copy()
        self.filtered_data = self.raw.get_data().copy()
        
        # Display settings
        self.epoch_length = self.DEFAULT_EPOCH
        self.current_time = 0.0
        self.y_scale = self.DEFAULT_SCALE
        
        # Channel-specific zoom levels (1.0 = default)
        self.channel_zooms = {ch: 1.0 for ch in self.raw.ch_names}
        
        # Setup plot
        self.setup_plot()
        
    def _load_data(self):
        """Load data based on file extension"""
        file_ext = os.path.splitext(self.filepath)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.raw = self._load_csv()
                self.logger.info("Loaded CSV file successfully")
            elif file_ext in ['.edf', '.set', '.fdt']:
                self.raw = self._load_mne()
                self.logger.info("Loaded MNE-compatible file successfully")
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            # Print data info
            self.logger.info(f"Channels: {self.raw.ch_names}")
            for ch in self.raw.ch_names:
                data = self.raw.get_data(picks=[ch])
                self.logger.info(f"{ch} - min: {data.min():.2f}, max: {data.max():.2f}, "
                               f"mean: {data.mean():.2f}, std: {data.std():.2f}")
                
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _load_csv(self):
        """Load data from CSV file"""
        self.logger.info("Reading CSV file...")
        data_dict = {
            'timestamp': [],
            'TP9': [], 'AF7': [], 'AF8': [], 'TP10': [],
            'AUX_L': [], 'AUX_R': []
        }
        
        valid_row_count = 0
        error_count = 0
        
        with open(self.filepath, 'r') as f:
            for i, line in enumerate(f):
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 6 and '/muse/eeg' in parts[1]:
                        # Convert timestamp
                        timestamp = float(parts[0])
                        
                        # Validate and convert EEG values
                        eeg_values = [float(val) for val in parts[2:8]]
                        
                        # Only add if all values are valid numbers
                        if all(not np.isnan(v) for v in eeg_values):
                            data_dict['timestamp'].append(timestamp)
                            data_dict['TP9'].append(eeg_values[0])
                            data_dict['AF7'].append(eeg_values[1])
                            data_dict['AF8'].append(eeg_values[2])
                            data_dict['TP10'].append(eeg_values[3])
                            data_dict['AUX_L'].append(eeg_values[4])
                            data_dict['AUX_R'].append(eeg_values[5])
                            valid_row_count += 1
                        else:
                            error_count += 1
                            
                except (ValueError, IndexError) as e:
                    error_count += 1
                    if error_count < 10:
                        self.logger.warning(f"Error parsing line {i}: {str(e)}")
                    continue
        
        self.logger.info(f"Processed {valid_row_count} valid rows with {error_count} errors")
        
        if valid_row_count == 0:
            raise ValueError("No valid data rows found in the CSV file")
        
        # Convert to numpy arrays
        ch_data = []
        ch_names = []
        
        # Process EEG channels first
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            if data_dict[ch]:
                ch_data.append(np.array(data_dict[ch]))
                ch_names.append(ch)
        
        # Process AUX channels
        aux_mapping = {'AUX_L': 'Left AUX', 'AUX_R': 'Right AUX'}
        for old_name, new_name in aux_mapping.items():
            if data_dict[old_name]:
                ch_data.append(np.array(data_dict[old_name]))
                ch_names.append(new_name)
        
        data_array = np.array(ch_data)
        self.logger.info(f"Created data array with shape: {data_array.shape}")
        
        # Handle NaN values
        if np.any(np.isnan(data_array)):
            self.logger.warning("NaN values detected, interpolating...")
            for i in range(len(ch_data)):
                mask = np.isnan(data_array[i])
                if np.any(mask):
                    data_array[i] = np.interp(
                        np.arange(len(data_array[i])),
                        np.arange(len(data_array[i]))[~mask],
                        data_array[i][~mask]
                    )
        
        # Create MNE info
        ch_types = ['eeg' if ch in self.CHANNELS['eeg'] else 'misc' for ch in ch_names]
        info = mne.create_info(ch_names=ch_names, sfreq=self.SF, ch_types=ch_types)
        
        return mne.io.RawArray(data_array, info)
        
    def _load_mne(self):
        """Load data using MNE"""
        try:
            if self.filepath.endswith('.set'):
                raw = mne.io.read_raw_eeglab(self.filepath, preload=True)
            else:
                raw = mne.io.read_raw_edf(self.filepath, preload=True)
            return raw
        except Exception as e:
            self.logger.error(f"Error loading MNE format: {str(e)}")
            raise
            
    def setup_plot(self):
            """Initialize the plot window and controls"""
            self.fig = plt.figure(figsize=(15, 10))
            
            # Create main plotting area with fixed space for controls
            self.fig.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.05, hspace=0.2)
            
            # Initialize axes for all channels using subplot directly instead of GridSpec
            self.channel_axes = {}
            all_channels = self.CHANNELS['eeg'] + self.CHANNELS['aux']
            n_channels = len(all_channels)
            
            for i, ch in enumerate(all_channels):
                ax = self.fig.add_subplot(n_channels, 1, i+1)
                self.channel_axes[ch] = ax
                ax.patch.set_alpha(0.0)
                ax._hover = False

            # Controls area - using figure coordinates
            right_margin = 0.98
            controls_width = right_margin - 0.88
            button_width = controls_width * 0.2  # 20% of control width
            button_height = 0.05
            slider_left = 0.88
            slider_width = controls_width

            # Epoch controls with title
            epoch_title_y = 0.9
            self.fig.text(0.93, epoch_title_y, f'Epoch = {self.DEFAULT_EPOCH}s', 
                        horizontalalignment='center', verticalalignment='center')
            self.epoch_title = self.fig.texts[-1]

            # Epoch slider directly under title
            epoch_slider_y = epoch_title_y - 0.05
            epoch_ax = plt.axes([slider_left, epoch_slider_y, slider_width, 0.02])
            self.epoch_slider = Slider(epoch_ax, '', 0.01, 60.0, valinit=self.DEFAULT_EPOCH)

            # Epoch buttons under slider
            button_y = epoch_slider_y - 0.07
            left_button_x = slider_left + (slider_width - 2*button_width)/2
            right_button_x = left_button_x + button_width
            self.epoch_left = Button(plt.axes([left_button_x, button_y, button_width, button_height]), '◂')
            self.epoch_right = Button(plt.axes([right_button_x, button_y, button_width, button_height]), '▸')

            # Scale controls
            scale_title_y = button_y - 0.1
            self.fig.text(0.93, scale_title_y, f'Scale = {self.DEFAULT_SCALE}μV', 
                        horizontalalignment='center', verticalalignment='center')
            self.scale_title = self.fig.texts[-1]

            # Scale slider directly under title
            scale_slider_y = scale_title_y - 0.05
            scale_ax = plt.axes([slider_left, scale_slider_y, slider_width, 0.02])
            self.scale_slider = Slider(scale_ax, '', 1, 1000, valinit=self.DEFAULT_SCALE)

            # Scale buttons under slider
            scale_button_y = scale_slider_y - 0.07
            self.scale_left = Button(plt.axes([left_button_x, scale_button_y, button_width, button_height]), '◂')
            self.scale_right = Button(plt.axes([right_button_x, scale_button_y, button_width, button_height]), '▸')

            # Reset button at bottom
            reset_y = scale_button_y - 0.1
            reset_ax = plt.axes([slider_left, reset_y, slider_width, button_height])
            self.reset_button = Button(reset_ax, 'Reset View')

            # Connect events
            self.epoch_slider.on_changed(self._on_epoch_change)
            self.epoch_left.on_clicked(lambda _: self._on_epoch_button(-0.1))
            self.epoch_right.on_clicked(lambda _: self._on_epoch_button(0.1))

            self.scale_slider.on_changed(self._on_scale_change)
            self.scale_left.on_clicked(lambda _: self._on_scale_button(-5))
            self.scale_right.on_clicked(lambda _: self._on_scale_button(5))
            self.reset_button.on_clicked(self._on_reset)
            
            # Connect mouse/keyboard events
            self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key)
            self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
            
            # Store original view limits
            self.original_view = {ax: ax.get_ylim() for ax in self.channel_axes.values()}
            
            self.update_plot()
            plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.05, hspace=0.2)

    def _on_epoch_text(self, val):
        """Handle epoch text input"""
        try:
            val = float(val)
            val = max(0.01, min(60.0, val))
            val = round(val * 10) / 10  # Round to nearest 0.1s
            self.epoch_length = val
            self.epoch_slider.set_val(val)
            self.epoch_text.set_val(f"{val:.1f}")
        except ValueError:
            self.epoch_text.set_val(f"{self.epoch_length:.1f}")

    def _on_scale_text(self, val):
        """Handle scale text input"""
        try:
            val = float(val)
            val = max(1, min(1000, val))
            val = round(val / 5) * 5  # Round to nearest 5
            self.y_scale = val
            self.scale_slider.set_val(val)
            self.scale_text.set_val(f"{val}")
        except ValueError:
            self.scale_text.set_val(f"{self.y_scale}")

    def _on_epoch_button(self, delta):
        """Handle epoch up/down button clicks"""
        new_val = round((self.epoch_length + delta) * 10) / 10  # Round to nearest 0.1
        new_val = max(0.01, min(60.0, new_val))
        self.epoch_slider.set_val(new_val)

    def _on_scale_button(self, delta):
        """Handle scale up/down button clicks"""
        new_val = round((self.y_scale + delta) / 5) * 5  # Round to nearest 5
        new_val = max(1, min(1000, new_val))
        self.scale_slider.set_val(new_val)

    def center_data(self):
        """Center the data around zero"""
        data = self.raw.get_data()
        for i, ch_name in enumerate(self.raw.ch_names):
            data[i, :] = data[i, :] - np.mean(data[i, :])
        self.raw = mne.io.RawArray(data, self.raw.info)

    def _on_epoch_change(self, val):
        """Handle epoch slider changes"""
        self.epoch_length = round(float(val) * 10) / 10  # Round to nearest 0.1s
        self.epoch_title.set_text(f'Epoch = {self.epoch_length}s')
        self.update_plot()

    def _on_scale_change(self, val):
            """Handle scale slider changes"""
            self.y_scale = round(float(val) / 5) * 5  # Round to nearest 5
            self.scale_title.set_text(f'Scale = {self.y_scale}μV')
            self.update_plot()
            
    def _on_reset(self, event):
        """Reset view to original state"""
        for ch, ax in self.channel_axes.items():
            ax.set_ylim(self.original_view[ax])
        self.channel_zooms = {ch: 1.0 for ch in self.raw.ch_names}
        self.epoch_length = self.DEFAULT_EPOCH
        self.y_scale = self.DEFAULT_SCALE
        self.epoch_slider.set_val(self.DEFAULT_EPOCH)
        self.scale_slider.set_val(self.DEFAULT_SCALE)
        self.update_plot()
        
    def _on_scroll(self, event):
        """Handle scroll events"""
        if event.inaxes and event.key == 'control':
            for ch, ax in self.channel_axes.items():
                if event.inaxes == ax:
                    zoom_factor = 1.1 if event.button == 'up' else 0.9
                    self.channel_zooms[ch] *= zoom_factor
                    self.update_plot()
                    break
                    
    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'left':
            self.current_time = max(0, self.current_time - self.epoch_length)
            self.update_plot()
        elif event.key == 'right':
            max_time = len(self.raw.times)/self.SF - self.epoch_length
            self.current_time = min(max_time, self.current_time + self.epoch_length)
            self.update_plot()
            
    def _on_hover(self, event):
        """Handle hover events"""
        for ch, ax in self.channel_axes.items():
            if event.inaxes == ax:
                if not ax._hover:
                    ax._hover = True
                    ax.patch.set_alpha(0.1)
                    ax.patch.set_facecolor('grey')
            else:
                if ax._hover:
                    ax._hover = False
                    ax.patch.set_alpha(0.0)
            ax.figure.canvas.draw_idle()
            
    def update_plot(self):
        """Update all plots"""
        start_samp = int(self.current_time * self.SF)
        end_samp = start_samp + int(self.epoch_length * self.SF)
        
        for ch, ax in self.channel_axes.items():
            ax.clear()
            
            # Get data for the channel
            ch_idx = self.raw.ch_names.index(ch)
            data = self.filtered_data[ch_idx, start_samp:end_samp]
            times = np.arange(start_samp, end_samp) / self.SF
            
            # Plot with offset and channel-specific zoom
            ax.plot(times, data, color=self.COLORS[ch], linewidth=0.5)
            
            # Configure axis
            channel_scale = self.y_scale / self.channel_zooms[ch]
            ax.set_xlim(self.current_time, self.current_time + self.epoch_length)
            ax.set_ylim(-channel_scale, channel_scale)
            ax.grid(True, which='major', linestyle='-', alpha=0.3)
            ax.grid(True, which='minor', linestyle=':', alpha=0.15)
            
            # Channel label
            ax.set_ylabel(f'{ch}\n(μV)')
            
            # Only show x-axis label for bottom plot
            if ch == self.CHANNELS['aux'][-1]:  # Last channel
                ax.set_xlabel('Time (MM:SS.ss)')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(
                    lambda x, _: f"{int(x//60):02d}:{x%60:05.2f}"))
            else:
                ax.set_xticklabels([])
            
            # Restore hover effect if active
            if hasattr(ax, '_hover') and ax._hover:
                ax.patch.set_alpha(0.1)
                ax.patch.set_facecolor('grey')
        
        # Update title
        self.fig.suptitle(f'Time Window: {self.current_time:.1f}s - '
                            f'{self.current_time + self.epoch_length:.1f}s')
        
        self.fig.canvas.draw_idle()

def main():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select data file",
        filetypes=[
            ("All supported", "*.csv;*.edf;*.set;*.fdt"),
            ("CSV files", "*.csv"),
            ("EDF files", "*.edf"),
            ("EEGLAB files", "*.set;*.fdt")
        ]
    )
    
    if file_path:
        try:
            viewer = MuseViewer(file_path)
            plt.show()
        except Exception as e:
            logging.error(f"Failed to initialize viewer: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    else:
        print("No file selected")

if __name__ == "__main__":
    main()