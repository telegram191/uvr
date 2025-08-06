import os,sys,torch,warnings,pdb
warnings.filterwarnings("ignore")
import librosa
import importlib
import numpy as np
import hashlib, math
from tqdm import tqdm
from uvr5_pack.lib_v5 import spec_utils
from uvr5_pack.utils import _get_name_params,inference
from uvr5_pack.lib_v5.model_param_init import ModelParameters
from scipy.io import wavfile
import multiprocessing as mp
from pathlib import Path
import glob
from pydub import AudioSegment
import tempfile

class _audio_pre_():
    def __init__(self, model_path,device,is_half):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            'postprocess': False,
            'tta': False,
            # Constants
            'window_size': 512,
            'agg': 10,
            'high_end_process': 'mirroring',
        }
        nn_arch_sizes = [
            31191, # default
            33966,61968, 123821, 123812, 537238 # custom
        ]
        self.nn_architecture = list('{}KB'.format(s) for s in nn_arch_sizes)
        model_size = math.ceil(os.stat(model_path ).st_size / 1024)
        nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x:abs(x-model_size)))
        nets = importlib.import_module('uvr5_pack.lib_v5.nets' + f'_{nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
        model_hash = hashlib.md5(open(model_path,'rb').read()).hexdigest()
        param_name ,model_params_d = _get_name_params(model_path , model_hash)

        mp = ModelParameters(model_params_d)
        model = nets.CascadedASPPNet(mp.param['bins'] * 2)
        cpk = torch.load( model_path , map_location='cpu')  
        model.load_state_dict(cpk)
        model.eval()
        if(is_half==True):model = model.half().to(device)
        else:model = model.to(device)

        self.mp = mp
        self.model = model

    def _path_audio_(self, music_file, output_root=None):
        if output_root is None:
            return "No output root."
            
        # Get base name without extension and create song directory
        name = os.path.splitext(os.path.basename(music_file))[0]
        song_dir = os.path.join(output_root, name)
        os.makedirs(song_dir, exist_ok=True)

        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param['band'])
        
        for d in range(bands_n, 0, -1): 
            bp = self.mp.param['band'][d]
            if d == bands_n: # high-end band
                X_wave[d] = librosa.load(
                    music_file, sr=bp['sr'], mono=False, dtype=np.float32, res_type=bp['res_type'])[0]
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], orig_sr=self.mp.param['band'][d+1]['sr'], target_sr=bp['sr'], res_type=bp['res_type'])
            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], self.mp.param['mid_side'], self.mp.param['mid_side_b2'], self.mp.param['reverse'])
            
            if d == bands_n and self.data['high_end_process'] != 'none':
                input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + ( self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data['agg']/100)
        aggressiveness = {'value': aggresive_set, 'split_bin': self.mp.param['band'][1]['crop_stop']}
        with torch.no_grad():
            pred, X_mag, X_phase = inference(X_spec_m,self.device,self.model, aggressiveness,self.data)
        
        if self.data['postprocess']:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        # Process instrument
        if self.data['high_end_process'].startswith('mirroring'):
            input_high_end_ = spec_utils.mirroring(self.data['high_end_process'], y_spec_m, input_high_end, self.mp)
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp,input_high_end_h, input_high_end_)
        else:
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
        
        # Process vocals
        if self.data['high_end_process'].startswith('mirroring'):
            input_high_end_ = spec_utils.mirroring(self.data['high_end_process'], v_spec_m, input_high_end, self.mp)
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp, input_high_end_h, input_high_end_)
        else:
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)

        # Create temporary directory in output folder
        temp_dir = os.path.join(song_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save temporary WAV files
            temp_inst = os.path.join(temp_dir, 'temp_inst.wav')
            temp_vocal = os.path.join(temp_dir, 'temp_vocal.wav')
            
            wavfile.write(temp_inst, self.mp.param['sr'], 
                         (np.array(wav_instrument)*32768).astype("int16"))
            wavfile.write(temp_vocal, self.mp.param['sr'], 
                         (np.array(wav_vocals)*32768).astype("int16"))
            
            # Convert to high quality MP3 using ffmpeg
            import subprocess
            
            # Convert instrument to MP3
            inst_output = os.path.join(song_dir, 'instrument.mp3')
            subprocess.run([
                'ffmpeg', '-y',  # Overwrite output files
                '-i', temp_inst,  # Input file
                '-acodec', 'libmp3lame',  # MP3 codec
                '-ab', '320k',  # Fixed bitrate
                '-ar', '44100',  # Sample rate
                '-joint_stereo', '1',  # Joint stereo
                inst_output
            ], check=True, capture_output=True)
            
            # Convert vocal to MP3
            vocal_output = os.path.join(song_dir, 'vocal.mp3')
            subprocess.run([
                'ffmpeg', '-y',  # Overwrite output files
                '-i', temp_vocal,  # Input file
                '-acodec', 'libmp3lame',  # MP3 codec
                '-ab', '320k',  # Fixed bitrate
                '-ar', '44100',  # Sample rate
                '-joint_stereo', '1',  # Joint stereo
                vocal_output
            ], check=True, capture_output=True)
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_inst):
                    os.remove(temp_inst)
                if os.path.exists(temp_vocal):
                    os.remove(temp_vocal)
                os.rmdir(temp_dir)
            except:
                pass  # Ignore cleanup errors
            
            print(f'Converting to MP3 with bitrate: 320k')

        print(f'{name} processing completed')

def read_processed_files(log_file):
    """Read the log file of processed files"""
    processed_files = set()
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    processed_files.add(line)
    return processed_files

def log_processed_file(log_file, file_path):
    """Log a processed file"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{file_path}\n")

def get_audio_files(input_folder, log_file):
    """Get all audio files from input folder that haven't been processed"""
    audio_extensions = ['*.mp3', '*.wav']
    audio_files = []
    
    # Get processed files from log
    processed_files = read_processed_files(log_file)
    
    # Get all audio files
    for ext in audio_extensions:
        found_files = glob.glob(os.path.join(input_folder, ext))
        for file in found_files:
            # Skip hidden files and temp files
            basename = os.path.basename(file)
            if not basename.startswith('.') and not basename.startswith('~'):
                # Convert to absolute path for consistent comparison
                abs_path = os.path.abspath(file)
                if abs_path not in processed_files:
                    audio_files.append(abs_path)
    
    # Sort files and remove duplicates
    audio_files = sorted(list(set(audio_files)))
    
    # Print found files for debugging
    if audio_files:
        print("\nFound new files to process:")
        for file in audio_files:
            print(f"  - {os.path.basename(file)} ({file})")
    else:
        print("\nNo new files to process.")
        if processed_files:
            print("\nPreviously processed files:")
            for file in sorted(processed_files):
                print(f"  - {os.path.basename(file)}")
    
    return audio_files

def process_audio_file(args):
    """Worker function for multiprocessing"""
    audio_file, model_path, device, is_half, output_root, log_file = args
    try:
        # Initialize model in each worker
        pre_fun = _audio_pre_(model_path=model_path, device=device, is_half=is_half)
        
        # Process the file
        pre_fun._path_audio_(audio_file, output_root)
        
        # Log successful processing
        log_processed_file(log_file, audio_file)
        
        return f"Successfully processed: {os.path.basename(audio_file)}"
    except Exception as e:
        error_msg = f"Error processing {os.path.basename(audio_file)}: {str(e)}"
        print(error_msg)  # Print error immediately
        return error_msg

if __name__ == '__main__':
    # Auto-detect device and number of workers
    import torch
    import multiprocessing as mp
    
    if torch.cuda.is_available():
        device = 'cuda'
        is_half = True
        num_workers = 1  # CUDA usually works better with 1 worker
    else:
        device = 'cpu'
        is_half = False
        num_workers = min(mp.cpu_count(), 3)  # Use up to 3 CPU cores
        print(f"CUDA not available, using CPU with {num_workers} workers")
    
    # Configuration
    model_path = 'uvr5_weights/2_HP-UVR.pth'
    input_folder = 'input'
    output_root = 'output'
    log_file = os.path.join('processed_files.log')
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' not found. Creating it...")
        os.makedirs(input_folder, exist_ok=True)
        print(f"Please put your audio files in the '{input_folder}' folder and run again.")
        sys.exit(1)
    
    # Create output folder
    os.makedirs(output_root, exist_ok=True)
    
    # Get all audio files that haven't been processed
    audio_files = get_audio_files(input_folder, log_file)
    
    if not audio_files:
        if os.path.exists(log_file):
            print("\nAll files have been processed.")
        else:
            print(f"\nNo audio files found in '{input_folder}' folder.")
            print("Supported formats: mp3, wav")
        sys.exit(0)
    
    print(f"\nProcessing {len(audio_files)} files with {num_workers} workers...")
    
    # Prepare arguments for multiprocessing
    args_list = [(audio_file, model_path, device, is_half, output_root, log_file) 
                 for audio_file in audio_files]
    
    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_audio_file, args_list),
            total=len(audio_files),
            desc="Processing files"
        ))
    
    # Print results
    print("\nProcessing completed!")
    for result in results:
        print(result)