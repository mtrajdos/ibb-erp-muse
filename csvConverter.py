import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import logging
from datetime import datetime
from scipy.io import savemat
import os

# Constants
SAMPLE_RATE = {
    'eeg': 256,
    'aux': 256
}

CHANNEL_CONFIGS = {
    'eeg': {
        'names': ['TP9', 'AF7', 'AF8', 'TP10'],
        'colors': ['blue', 'red', 'green', 'purple'],
        'range': [0, 1700.0000]  # μV
    },
    'aux': {
        'names': ['AUX_L', 'AUX_R'],
        'range': [0, 1700.0000]    # μV
    }
}

class MuseProcessor:
    def __init__(self):
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        self.setup_logging()
        self.eeg_data = None
        self.aux_data = None
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = logging.getLogger('muse_processor')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_dir / f'processing_{timestamp}.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def load_muse_data(self, filepath):
        """Load raw Muse data."""
        self.logger.info(f"Loading Muse data from {filepath}")
        
        eeg_data = {
            'timestamp': [],
            'TP9': [], 'AF7': [], 'AF8': [], 'TP10': [],
            'AUX_L': [], 'AUX_R': []
        }
        
        first_timestamp = None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    parts = line.strip().split(',')
                    timestamp = float(parts[0])
                    
                    if '/muse/eeg' in parts[1]:
                        if first_timestamp is None:
                            first_timestamp = timestamp
                        
                        eeg_data['timestamp'].append(timestamp)
                        eeg_data['TP9'].append(float(parts[2]))
                        eeg_data['AF7'].append(float(parts[3]))
                        eeg_data['AF8'].append(float(parts[4]))
                        eeg_data['TP10'].append(float(parts[5]))
                        eeg_data['AUX_L'].append(float(parts[6]))
                        eeg_data['AUX_R'].append(float(parts[7]) if len(parts) > 7 else float('nan'))
                            
                except Exception as e:
                    self.logger.error(f"Error processing line {i}: {str(e)}")
                    continue
        
        self.eeg_data = pd.DataFrame(eeg_data)
        self.eeg_data['Time'] = self.eeg_data['timestamp'] - first_timestamp
        
        # Create separate AUX data
        self.aux_data = pd.DataFrame({
            'Time': self.eeg_data['Time'],
            'AUX_L': self.eeg_data['AUX_L'],
            'AUX_R': self.eeg_data['AUX_R']
        })
        
        return True

    def get_raw_mne(self):
        """Convert current EEG data to MNE Raw object."""
        ch_names = CHANNEL_CONFIGS['eeg']['names']
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=SAMPLE_RATE['eeg'], ch_types=ch_types)
        
        eeg_data = self.eeg_data[ch_names].values.T
        raw = mne.io.RawArray(eeg_data, info)
        
        return raw
            
    def process_eeg_data(self):
        """Apply proper filtering to EEG data."""
        self.logger.info("Processing EEG data with filters")
        
        # Get raw MNE object
        raw = self.get_raw_mne()
        
        # Apply filters
        raw.filter(l_freq=0.5, h_freq=50, method='iir', 
                iir_params=dict(order=1, ftype="butter"), verbose=True)
        raw.notch_filter(freqs=50, picks='eeg', method='fir', verbose=True)
        
        # Convert back to DataFrame
        filtered_data = raw.get_data()
        for i, ch in enumerate(CHANNEL_CONFIGS['eeg']['names']):
            self.eeg_data[ch] = filtered_data[i]
        
        self.logger.info("Finished applying filters to EEG data")
        return raw

    def export_to_eeglab(self, raw, filename):
        """Export to EEGLAB format compatible with Wonambi."""
        data = raw.get_data()
        if data is None:
            raise ValueError("No data available for export")

        # Create base filenames
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        fdt_filename = base_filename + '.fdt'

        eeglab_data = {
            'EEG': {
                'setname': base_filename,
                'filename': base_filename + '.set',
                'filepath': os.path.dirname(filename),
                'subject': '',
                'group': '',
                'condition': '',
                'session': 1,
                'comments': 'Exported with muse_processor',
                'nbchan': len(raw.ch_names),
                'trials': 1,
                'pnts': raw.n_times,
                'srate': raw.info['sfreq'],
                'xmin': 0,
                'xmax': (raw.n_times - 1) / raw.info['sfreq'],
                'times': np.arange(raw.n_times) / raw.info['sfreq'],
                'data': 'in_fdt_file',
                'datfile': fdt_filename,
                'chanlocs': [],
                'urchanlocs': [],
                'chaninfo': {'plotrad': [], 'shrink': [], 'nosedir': '+X', 'nodatchans': []},
                'ref': 'common',
                'event': np.array([]),
                'urevent': np.array([]),
                'epochdescription': {},
                'reject': {},
                'stats': {},
                'specdata': {},
                'specicaact': {},
                'splinefile': '',
                'icascalp': {},
                'icawinv': np.array([]),
                'icasphere': np.array([]),
                'icaweights': np.array([]),
                'icaerp': {},
                'icaact': np.array([]),
                'saved': 'yes'
            }
        }

        # Define channel types
        chan_types = {
            'TP9': 'EEG',
            'TP10': 'EEG',
            'AF7': 'EEG',
            'AF8': 'EEG',
            'AUX_L': 'AUX',
            'AUX_R': 'AUX'
        }

        # Add channel information with proper types
        for idx, ch_name in enumerate(raw.ch_names):
            chan_info = {
                'labels': str(ch_name),
                'type': chan_types.get(ch_name, 'EEG'),
                'unit': 'µV',
                'ref': 'common',
                'theta': float(idx * (360 / len(raw.ch_names))),
                'radius': 0.85,
                'X': 0,
                'Y': 0,
                'Z': 0,
                'sph_theta': 0,
                'sph_phi': 0,
                'sph_radius': 1,
                'urchan': idx + 1
            }
            eeglab_data['EEG']['chanlocs'].append(chan_info)
            eeglab_data['EEG']['urchanlocs'].append(chan_info)

        # Save .set file
        savemat(filename, eeglab_data, appendmat=False)

        # Save .fdt file with correct data type and ordering
        data = data.astype(np.float32)
        data = np.asfortranarray(data)
        fdt_file = os.path.splitext(filename)[0] + '.fdt'
        data.tofile(fdt_file)

        self.logger.info(f"Exported data to {filename} and {fdt_file}")

def main():
    try:
        processor = MuseProcessor()
        
        # Select file
        root = tk.Tk()
        root.withdraw()
        
        print("\nPlease select the Muse data file...")
        muse_file = filedialog.askopenfilename(
            title="Select Muse data file",
            filetypes=[("CSV files", "*.csv")]
        )
        if not muse_file:
            print("No file selected. Exiting...")
            return
            
        # Load data
        print("\nLoading data...")
        if not processor.load_muse_data(muse_file):
            print("Failed to load data. Exiting...")
            return
        
        # Get output directory and base filename
        output_dir = processor.output_dir
        base_name = os.path.splitext(os.path.basename(muse_file))[0]
        
        # Export raw data
        print("\nExporting raw data...")
        raw_unfiltered = processor.get_raw_mne()
        processor.export_to_eeglab(raw_unfiltered, 
                                 output_dir / f"{base_name}_raw.set")
        
        # Process and export filtered data
        print("\nApplying filters...")
        raw_filtered = processor.process_eeg_data()
        
        print("\nExporting filtered data...")
        processor.export_to_eeglab(raw_filtered, 
                                 output_dir / f"{base_name}_filtered.set")
        
        print(f"\nData processed and saved to: {output_dir}")
        print("\nFiles created:")
        print(f"  - {base_name}_raw.set/.fdt")
        print(f"  - {base_name}_filtered.set/.fdt")
        
        print("\nProcessing complete!")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()