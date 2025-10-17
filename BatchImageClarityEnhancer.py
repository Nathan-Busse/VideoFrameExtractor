# ==============================================================================
# GPU ACCELERATION PROTOCOL (NVIDIA/CUDA)
#
# This script is configured to use the fastest possible execution path:
# 1. Python Multiprocessing (true parallel processing across CPU cores).
# 2. OpenCV DNN CUDA Backend (GPU acceleration).
#
# If the script runs slowly or throws an error (e.g., "Requested backend/target is not supported"),
# it means the GPU environment is not properly configured. Follow these steps:
#
# ------------------------------------------------------------------------------
# STEP 1: Confirm NVIDIA Drivers and Hardware
# ------------------------------------------------------------------------------
# - Ensure you have a CUDA-compatible NVIDIA GPU.
# - Install the latest stable NVIDIA Graphics Drivers from the official website.
#
# ------------------------------------------------------------------------------
# STEP 2: Install the CUDA Toolkit
# ------------------------------------------------------------------------------
# - CRITICAL: Download and install a version of the CUDA Toolkit that is
#   compatible with the OpenCV version you are using. Version matching is key.
#   Start by checking common versions like CUDA 11.8 or 12.1.
# - Download from the NVIDIA CUDA Toolkit Archive.
#
# ------------------------------------------------------------------------------
# STEP 3: Install cuDNN (Deep Neural Network Library)
# ------------------------------------------------------------------------------
# - Download the cuDNN library version that matches your installed CUDA Toolkit.
# - This is a manual installation: Unzip the cuDNN archive and copy the files
#   (from the 'bin', 'include', and 'lib' folders) into the corresponding
#   subdirectories of your CUDA Toolkit installation path.
#
# ------------------------------------------------------------------------------
# STEP 4: Code Activation (Already Implemented Below)
# ------------------------------------------------------------------------------
# - The lines below explicitly tell OpenCV to use the CUDA backend and target:
#   sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#   sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# ------------------------------------------------------------------------------
#
# If the GPU setup fails, temporarily comment out the two CUDA lines above and
# uncomment the two CPU lines, or switch to the much faster ESPCN model.
# ==============================================================================

import cv2
import os
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from multiprocessing import Pool, Queue
import time
import queue as q_temp # Using a different alias for the standard queue
import threading

# --- GLOBAL SETTINGS ---
# Use all cores minus one to keep the system responsive
CPU_CORES = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

class BatchImageClarityEnhancer:
    def __init__(self, master):
        self.master = master
        master.title("Batch Image Clarity Enhancer (GPU/Multiprocessing)")
        master.geometry("800x600")
        master.configure(bg="#2c3e50")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready. Select directories and model.")

        self.is_running = False
        self.job_queue = []
        self.total_frames = 0
        self.processed_count = 0
        self.stop_event = threading.Event()

        # Multiprocessing Queue for progress updates from worker processes
        self.progress_queue = Queue()

        self.create_widgets()

    def create_widgets(self):
        # Apply a modern style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#34495e')
        style.configure('TLabel', background='#2c3e50', foreground='white', font=('Inter', 10))
        style.configure('TButton', font=('Inter', 10, 'bold'), padding=6, background='#1abc9c', foreground='white')
        style.map('TButton', background=[('active', '#16a085')])
        style.configure('TProgressbar', thickness=15, troughcolor='#34495e', background='#2ecc71')

        main_frame = ttk.Frame(self.master, padding="15", style='TFrame')
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Super Resolution Batch Processor", font=('Inter', 18, 'bold'), anchor='center')
        title_label.pack(pady=(5, 20))

        # --- Directory Selection Frame ---
        dir_frame = ttk.Frame(main_frame, style='TFrame')
        dir_frame.pack(fill='x', pady=5)

        # Input Directory
        self.create_path_selector(dir_frame, "Input Frames Directory:", self.input_dir, self.select_input_dir)
        # Output Directory
        self.create_path_selector(dir_frame, "Output Directory:", self.output_dir, self.select_output_dir)
        # Model File
        self.create_path_selector(dir_frame, "Model (.pb) File:", self.model_path, self.select_model_file)

        # --- Status and Progress ---
        status_label = ttk.Label(main_frame, text="Status:", style='TLabel')
        status_label.pack(fill='x', pady=(10, 0))

        status_display = ttk.Label(main_frame, textvariable=self.status_var, background='#34495e', foreground='#f1c40f', font=('Inter', 10, 'italic'), padding=5)
        status_display.pack(fill='x', pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="determinate", style='TProgressbar')
        self.progress_bar.pack(fill='x', pady=10)

        # --- Control Buttons ---
        button_frame = ttk.Frame(main_frame, style='TFrame')
        button_frame.pack(fill='x', pady=20)

        self.start_button = ttk.Button(button_frame, text="Start Enhancement", command=self.start_enhancement, style='TButton')
        self.start_button.pack(side='left', expand=True, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_enhancement, state='disabled', style='TButton')
        self.stop_button.pack(side='left', expand=True, padx=10)

        # --- Log Output ---
        log_label = ttk.Label(main_frame, text="Processing Log:", style='TLabel')
        log_label.pack(fill='x', pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(main_frame, height=8, state='disabled', bg='#1f2b37', fg='#ecf0f1', font=('Inter', 9), relief=tk.FLAT)
        self.log_text.pack(fill='both', expand=True)

        # Show CPU/Core Info
        core_info = ttk.Label(main_frame, text=f"Configured to use {CPU_CORES} CPU/Multiprocessing Cores.", background='#2c3e50', foreground='#95a5a6', font=('Inter', 8))
        core_info.pack(pady=5)


    def create_path_selector(self, parent, label_text, var, command):
        frame = ttk.Frame(parent, style='TFrame')
        frame.pack(fill='x', pady=5)

        label = ttk.Label(frame, text=label_text, width=25)
        label.pack(side='left', padx=(0, 10))

        entry = ttk.Entry(frame, textvariable=var, state='readonly', width=60)
        entry.pack(side='left', fill='x', expand=True)

        button = ttk.Button(frame, text="Browse", command=command)
        button.pack(side='right', padx=5)

    def log(self, message):
        self.master.after(0, self._append_log, message)

    def _append_log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.yview(tk.END)
        self.log_text.config(state='disabled')

    # --- Directory and File Selectors ---
    def select_input_dir(self):
        directory = filedialog.askdirectory(title="Select Directory with Input Frames")
        if directory:
            self.input_dir.set(directory)

    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory for Enhanced Frames")
        if directory:
            self.output_dir.set(directory)

    def select_model_file(self):
        file = filedialog.askopenfilename(title="Select Super Resolution Model (.pb)", filetypes=[("Protocol Buffer Models", "*.pb")])
        if file:
            self.model_path.set(file)

    # --- Enhancement Logic ---

    def start_enhancement(self):
        if self.is_running:
            self.status_var.set("Already running...")
            return

        input_path = self.input_dir.get()
        output_path = self.output_dir.get()
        model_file = self.model_path.get()

        if not all([input_path, output_path, model_file]):
            self.status_var.set("Error: All paths (Input, Output, Model) must be selected.")
            return

        if not os.path.exists(model_file):
            self.status_var.set(f"Error: Model file not found at {model_file}")
            return

        # 1. Prepare Job List
        image_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            self.status_var.set("Error: No images found in the input directory.")
            return

        self.job_queue = [(os.path.join(input_path, f), os.path.join(output_path, f), model_file) for f in image_files]
        self.total_frames = len(self.job_queue)
        self.processed_count = 0
        self.stop_event.clear()
        self.is_running = True
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')

        self.status_var.set(f"Starting enhancement on {self.total_frames} frames using {CPU_CORES} cores...")
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = self.total_frames

        # 2. Start Multiprocessing Pool in a separate thread
        threading.Thread(target=self._run_multiprocessing_pool, daemon=True).start()

        # 3. Start UI progress monitoring
        self.master.after(100, self.update_progress)


    def _run_multiprocessing_pool(self):
        # We use 'with Pool' for safe process management
        try:
            with Pool(processes=CPU_CORES) as pool:
                # Map the process_frame_worker function to the job_queue
                # We pass the progress_queue as a constant argument for communication
                pool.starmap(self._process_frame_worker, [(task + (self.progress_queue, self.stop_event)) for task in self.job_queue])
        except Exception as e:
            self.log(f"Multiprocessing Error: {e}")
        finally:
            self.is_running = False
            self.master.after(0, self._finish_enhancement)

    @staticmethod
    def _process_frame_worker(input_path, output_path, model_file, progress_queue, stop_event):
        if stop_event.is_set():
            return

        try:
            # 1. Load the Super Resolution model
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_file)

            # Extract scale from filename (e.g., LapSRN_x8.pb -> 8)
            scale = int(model_file.split('_x')[-1].split('.')[0])
            sr.setModel("lapsrn", scale)

            # 2. Configure Backend for Maximum Speed (CUDA/GPU)
            # This is the essential step for performance.
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            # --- Fallback Option (If CUDA setup fails, use this instead) ---
            # sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            # sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # 3. Read the image
            img = cv2.imread(input_path)
            if img is None:
                progress_queue.put(f"FAIL: Could not read image {os.path.basename(input_path)}")
                return

            # 4. Perform Upscaling
            enhanced_img = sr.upsample(img)

            # 5. Write the enhanced image
            cv2.imwrite(output_path, enhanced_img)

            # 6. Report Success
            progress_queue.put(f"SUCCESS: {os.path.basename(input_path)}")

        except Exception as e:
            # Report failure
            progress_queue.put(f"FAIL: {os.path.basename(input_path)} - {e}")

    def update_progress(self):
        # Safely pull messages from the Multiprocessing Queue
        while True:
            try:
                # Use standard queue to safely get from the multiprocessing queue
                message = self.progress_queue.get_nowait()
                self.processed_count += 1

                if message.startswith("SUCCESS"):
                    self.log(message)
                else:
                    # Log errors (e.g., failed read, CUDA failure, etc.)
                    self.log(f"ERROR: {message}")

                # Update UI elements
                progress = (self.processed_count / self.total_frames) * 100
                self.progress_bar['value'] = self.processed_count
                self.status_var.set(f"Processing... {self.processed_count} of {self.total_frames} frames ({progress:.1f}%)")

            except q_temp.Empty:
                # No more items in the queue for now
                break

        if self.is_running:
            # Schedule the next check
            self.master.after(100, self.update_progress)

    def stop_enhancement(self):
        self.stop_event.set()
        self.is_running = False
        self.status_var.set("Stopping workers. Please wait...")
        self._finish_enhancement()

    def _finish_enhancement(self):
        # Called when the pool finishes or stop is clicked
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        if not self.stop_event.is_set():
            self.status_var.set(f"Finished! {self.processed_count} frames processed.")
        else:
            self.status_var.set("Stopped by user.")


if __name__ == '__main__':
    # Initialize necessary Firebase variables for Canvas execution environment (not used by this local script but good practice)
    # The cv2 import must be done before the app loop starts
    try:
        root = tk.Tk()
        app = BatchImageClarityEnhancer(root)
        root.mainloop()
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")

