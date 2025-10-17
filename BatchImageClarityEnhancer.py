import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import multiprocessing
import time
import math
from functools import partial

# --- GLOBAL APPLICATION VARIABLES ---
# These are managed in the main process
model_path = ""
input_folder = ""
output_folder = ""
is_processing = False
total_files = 0
processed_count = 0
manager = None
status_queue = None
pool = None
root = None

def process_frame_worker(file_name, input_dir, output_dir, model_path_str, status_q):
    """
    Worker function executed by a separate process.
    It loads the model, processes a single frame, and reports status via a Queue.
    
    CRITICAL: Configured to use CUDA/GPU. If the environment is not set up 
    with CUDA Toolkit and cuDNN, this will throw an error and revert to 
    the previous CPU configuration is recommended.
    """
    try:
        # 1. Read the image
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"enhanced_{file_name}")
        
        frame = cv2.imread(input_path)
        if frame is None:
            status_q.put({"type": "error", "file": file_name, "message": "Could not read image file."})
            return 1 # Return 1 for failure, 0 for success

        # 2. Initialize and Load Model (Must be done per process as sr object may not be pickleable)
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        
        # Determine model name and scale from the filename (e.g., 'LapSRN_x8.pb')
        model_name = os.path.basename(model_path_str).split('_')[0].lower()
        scale = int(os.path.basename(model_path_str).split('x')[-1].split('.')[0])
        
        sr.readModel(model_path_str)
        sr.setModel(model_name, scale)

        # --- FINAL GPU CONFIGURATION ---
        # Explicitly setting to CUDA backend for high-speed GPU processing
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # 3. Perform Super Resolution
        result = sr.upsample(frame)
        
        # 4. Save the result
        cv2.imwrite(output_path, result)
        
        # 5. Report success
        status_q.put({"type": "progress", "file": file_name})
        return 0

    except cv2.error as e:
        status_q.put({"type": "error", "file": file_name, "message": f"OpenCV Error: {e}"})
        return 1
    except Exception as e:
        status_q.put({"type": "error", "file": file_name, "message": f"Unexpected Error: {e}"})
        return 1
        
# --- GUI and Main Thread Logic ---

def create_gui():
    """Initializes the main application window."""
    global root, status_var, progress_bar, start_button, stop_button

    root = tk.Tk()
    root.title("Batch Image Clarity Enhancer (GPU/Multiprocessing)")
    root.geometry("800x500")
    root.configure(bg='#2e2e2e')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TFrame', background='#2e2e2e')
    style.configure('TButton', background='#4a4a4a', foreground='white', font=('Inter', 10, 'bold'), borderwidth=1)
    style.map('TButton', background=[('active', '#5c5c5c')])
    style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Inter', 10))
    style.configure('TProgressbar', background='#00c04b', troughcolor='#3c3c3c') # Green bar to signify speed

    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.pack(fill='both', expand=True)

    # --- INPUT/OUTPUT PATHS ---
    path_frame = ttk.Frame(main_frame)
    path_frame.pack(fill='x', pady=10)

    ttk.Label(path_frame, text="1. Input Frames Folder:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
    global input_path_label
    input_path_label = ttk.Label(path_frame, text="No folder selected", background='#3c3c3c', foreground='#a0a0a0', anchor='w', relief='groove')
    input_path_label.grid(row=0, column=1, sticky='ew', padx=5, pady=5, ipadx=5, ipady=5)
    ttk.Button(path_frame, text="Browse", command=select_input_folder).grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(path_frame, text="2. Output Enhanced Folder:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
    global output_path_label
    output_path_label = ttk.Label(path_frame, text="No folder selected", background='#3c3c3c', foreground='#a0a0a0', anchor='w', relief='groove')
    output_path_label.grid(row=1, column=1, sticky='ew', padx=5, pady=5, ipadx=5, ipady=5)
    ttk.Button(path_frame, text="Browse", command=select_output_folder).grid(row=1, column=2, padx=5, pady=5)
    
    path_frame.grid_columnconfigure(1, weight=1)

    # --- MODEL SELECTION ---
    model_frame = ttk.Frame(main_frame)
    model_frame.pack(fill='x', pady=10)

    ttk.Label(model_frame, text="3. Model File (.pb):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
    global model_path_label
    model_path_label = ttk.Label(model_frame, text="No model selected", background='#3c3c3c', foreground='#a0a0a0', anchor='w', relief='groove')
    model_path_label.grid(row=0, column=1, sticky='ew', padx=5, pady=5, ipadx=5, ipady=5)
    ttk.Button(model_frame, text="Browse", command=select_model_file).grid(row=0, column=2, padx=5, pady=5)

    model_frame.grid_columnconfigure(1, weight=1)

    # --- CONTROL BUTTONS ---
    control_frame = ttk.Frame(main_frame)
    control_frame.pack(fill='x', pady=20)

    start_button = ttk.Button(control_frame, text="Start Enhancement (GPU Accelerated)", command=start_processing, style='TButton')
    start_button.pack(side='left', fill='x', expand=True, padx=5)

    stop_button = ttk.Button(control_frame, text="Stop Processing", command=stop_processing, state=tk.DISABLED, style='TButton')
    stop_button.pack(side='left', fill='x', expand=True, padx=5)

    # --- STATUS AND PROGRESS ---
    global progress_bar
    progress_bar = ttk.Progressbar(main_frame, orient='horizontal', length=760, mode='determinate')
    progress_bar.pack(fill='x', pady=10)

    global status_var
    status_var = tk.StringVar(value="Ready. Requires CUDA/cuDNN environment setup.")
    ttk.Label(main_frame, textvariable=status_var, font=('Inter', 10, 'italic'), foreground='#00c04b').pack(fill='x', pady=5)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()

def select_input_folder():
    """Opens a dialog to select the folder containing low-res frames."""
    global input_folder
    folder = filedialog.askdirectory(title="Select Input Frames Folder")
    if folder:
        input_folder = folder
        input_path_label.config(text=input_folder, foreground='white')

def select_output_folder():
    """Opens a dialog to select the folder for enhanced frames."""
    global output_folder
    folder = filedialog.askdirectory(title="Select Output Enhanced Folder")
    if folder:
        output_folder = folder
        output_path_label.config(text=output_folder, foreground='white')
        
def select_model_file():
    """Opens a dialog to select the Super Resolution model file (.pb)."""
    global model_path
    file = filedialog.askopenfilename(
        title="Select Super Resolution Model File (.pb)",
        filetypes=(("OpenCV Model Files", "*.pb"), ("All files", "*.*"))
    )
    if file:
        model_path = file
        model_path_label.config(text=os.path.basename(model_path), foreground='white')

def start_processing():
    """Checks prerequisites, initializes the pool, and starts the processing thread."""
    global input_folder, output_folder, model_path, is_processing, total_files, processed_count, manager, status_queue, pool

    if not input_folder or not output_folder or not model_path:
        messagebox.showerror("Missing Information", "Please select input folder, output folder, and the model file (.pb).")
        return

    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_files = len(input_files)
    processed_count = 0

    if total_files == 0:
        messagebox.showinfo("No Files", "Input folder is empty or contains no supported image files.")
        return

    # Initialize multiprocessing components
    manager = multiprocessing.Manager()
    status_queue = manager.Queue()
    
    # Use all available cores minus 1 for the GUI responsiveness
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Prepare the partial function for the pool to receive fixed args
    process_func = partial(
        process_frame_worker, 
        input_dir=input_folder, 
        output_dir=output_folder, 
        model_path_str=model_path, 
        status_q=status_queue
    )
    
    # Disable start, enable stop button
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    is_processing = True
    progress_bar['value'] = 0
    status_var.set(f"Starting GPU-accelerated pool on {num_cores} workers. Total files: {total_files}")
    
    # Start all tasks asynchronously
    global async_result
    # The pool handles the scheduling of image files to workers (processes)
    async_result = pool.map_async(process_func, input_files)

    # Start polling the queue for updates
    root.after(100, update_progress)

def update_progress():
    """Polls the status queue and updates the GUI progress bar and status label."""
    global is_processing, processed_count, total_files, async_result

    if not is_processing:
        return

    while not status_queue.empty():
        item = status_queue.get()
        if item["type"] == "progress":
            processed_count += 1
            progress = (processed_count / total_files) * 100
            progress_bar['value'] = progress
            status_var.set(f"Processing ({processed_count}/{total_files}): {item['file']}")
        elif item["type"] == "error":
            # An error occurred in a worker, log it but don't stop the whole process
            print(f"Error processing {item['file']}: {item['message']}")
            # We still need to count this as 'processed' to update the progress bar correctly
            processed_count += 1 
            progress = (processed_count / total_files) * 100
            progress_bar['value'] = progress
            status_var.set(f"Error on {item['file']}. Continuing...")
    
    # Check if all processes are finished
    if async_result.ready():
        # Clean up and finalize
        finalize_processing(True)
        return

    # Re-schedule the progress check
    root.after(100, update_progress)


def stop_processing():
    """Sets the flag to stop the processing worker gracefully and terminates the pool."""
    global is_processing, pool
    
    if pool:
        pool.terminate()
        pool.join() # Wait for the worker processes to exit
        
    finalize_processing(False)
        
def finalize_processing(is_completed):
    """Resets the UI state after the batch finishes or is stopped."""
    global is_processing, pool, processed_count, total_files
    
    is_processing = False
    
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    
    if is_completed:
        final_message = f"Batch enhancement completed! {processed_count} files saved to: {output_folder}"
        status_var.set(final_message)
        messagebox.showinfo("Process Complete", final_message)
    else:
        final_message = f"Processing stopped by user. {processed_count} files enhanced so far."
        status_var.set(final_message)
        messagebox.showwarning("Process Stopped", final_message)

    if pool:
        pool.close()
        pool.join()
        pool = None

def on_closing():
    """Handles window closing by stopping the processes."""
    global is_processing
    if is_processing:
        stop_processing()
    root.destroy()


if __name__ == '__main__':
    # Add this line for Windows compatibility when using multiprocessing
    multiprocessing.freeze_support()
    create_gui()

