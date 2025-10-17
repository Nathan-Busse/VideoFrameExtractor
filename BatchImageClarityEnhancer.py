import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import time
from PIL import Image, ImageTk

# --- GLOBAL APPLICATION VARIABLES ---
# Placeholder for the loaded super resolution model
model_path = ""
input_folder = ""
output_folder = ""

# Status tracking
is_processing = False
# Lock for updating the status label from the worker thread
status_lock = threading.Lock()

def create_gui():
    """Initializes the main application window."""
    global root, status_var, progress_bar, start_button, stop_button

    root = tk.Tk()
    root.title("Batch Image Clarity Enhancer (OpenCV/DNN)")
    root.geometry("800x500")
    root.configure(bg='#2e2e2e')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TFrame', background='#2e2e2e')
    style.configure('TButton', background='#4a4a4a', foreground='white', font=('Inter', 10, 'bold'), borderwidth=1)
    style.map('TButton', background=[('active', '#5c5c5c')])
    style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Inter', 10))
    style.configure('TProgressbar', background='#007acc', troughcolor='#3c3c3c')

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

    start_button = ttk.Button(control_frame, text="Start Enhancement", command=start_processing_thread, style='TButton')
    start_button.pack(side='left', fill='x', expand=True, padx=5)

    stop_button = ttk.Button(control_frame, text="Stop Processing", command=stop_processing, state=tk.DISABLED, style='TButton')
    stop_button.pack(side='left', fill='x', expand=True, padx=5)

    # --- STATUS AND PROGRESS ---
    progress_bar = ttk.Progressbar(main_frame, orient='horizontal', length=760, mode='determinate')
    progress_bar.pack(fill='x', pady=10)

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(main_frame, textvariable=status_var, font=('Inter', 10, 'italic'), foreground='#00c04b').pack(fill='x', pady=5)

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

def start_processing_thread():
    """Checks prerequisites and starts the enhancement thread."""
    global input_folder, output_folder, model_path, is_processing

    if not input_folder or not output_folder or not model_path:
        messagebox.showerror("Missing Information", "Please select input folder, output folder, and the model file (.pb).")
        return

    if is_processing:
        messagebox.showinfo("Processing in Progress", "The batch enhancement is already running.")
        return

    # Disable start and enable stop buttons
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    is_processing = True
    
    # Reset progress bar and status
    progress_bar['value'] = 0
    with status_lock:
        status_var.set("Processing started...")

    # Start the worker thread
    thread = threading.Thread(target=batch_enhancement_worker)
    thread.daemon = True
    thread.start()

def stop_processing():
    """Sets the flag to stop the processing worker gracefully."""
    global is_processing
    is_processing = False
    stop_button.config(state=tk.DISABLED)
    with status_lock:
        status_var.set("Stopping at the next frame...")
        
def batch_enhancement_worker():
    """Worker function that runs the batch enhancement process."""
    global is_processing, input_folder, output_folder, model_path
    
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_files = len(input_files)
    processed_count = 0
    
    if total_files == 0:
        with status_lock:
            status_var.set("Input folder is empty or contains no supported image files.")
        finalize_processing()
        return

    # --- MODEL INITIALIZATION ---
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        
        model_name = os.path.basename(model_path).split('_')[0].lower()
        scale = int(os.path.basename(model_path).split('x')[-1].split('.')[0])
        
        with status_lock:
             status_var.set(f"Loading model: {model_name} at {scale}x scale...")
             
        sr.readModel(model_path)
        sr.setModel(model_name, scale)
        
        # ----------------------------------------------------------------------
        # --- CRITICAL FIX: CUDA ASSERTION ERROR (Setting CPU Fallback) ---
        # ----------------------------------------------------------------------
        # This fixes the error: preferableBackend != DNN_BACKEND_CUDA
        # We explicitly set the backend to OPENCV and the target to CPU.
        # This will run on your CPU, bypassing the missing CUDA build support.
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with status_lock:
            status_var.set(f"Model loaded. Processing {total_files} files using CPU fallback (Slower but stable)...")

    except AttributeError:
        # This occurs if dnn_superres wasn't found (if user forgot opencv-contrib-python install)
        with status_lock:
            status_var.set("Error: cv2.dnn_superres not found. Please ensure 'opencv-contrib-python' is installed.")
        finalize_processing()
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        with status_lock:
            status_var.set(f"Error loading model: Check console for details.")
        finalize_processing()
        return

    # --- BATCH PROCESSING LOOP ---
    for file_name in input_files:
        if not is_processing:
            break
            
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f"enhanced_{file_name}")
        
        try:
            # 1. Read the image
            frame = cv2.imread(input_path)
            if frame is None:
                print(f"Warning: Could not read image file {file_name}. Skipping.")
                continue

            # 2. Perform Super Resolution
            with status_lock:
                status_var.set(f"Enhancing ({processed_count + 1}/{total_files}): {file_name}")
                
            # Perform upscaling
            result = sr.upsample(frame)
            
            # 3. Save the result
            cv2.imwrite(output_path, result)
            
            processed_count += 1
            progress_bar['value'] = (processed_count / total_files) * 100
            
        except cv2.error as e:
            # Catch file-specific OpenCV errors
            print(f"Error processing {input_path}: {e}")
            with status_lock:
                status_var.set(f"Error processing {file_name}. Check console for details.")
            # Do not stop processing, skip to the next file
            
        except Exception as e:
            # Catch other unexpected Python errors
            print(f"An unexpected error occurred processing {input_path}: {e}")
            with status_lock:
                status_var.set(f"An unexpected error occurred. Check console.")
            # Do not stop processing, skip to the next file


    # --- FINALIZATION ---
    finalize_processing()
    
    if processed_count == total_files:
        final_message = f"Batch enhancement completed! {processed_count} files saved to: {output_folder}"
    elif processed_count > 0:
        final_message = f"Processing stopped by user. {processed_count} files enhanced."
    else:
        final_message = "Processing stopped or failed before any files were enhanced."
        
    with status_lock:
        status_var.set(final_message)
        messagebox.showinfo("Process Complete", final_message)

def finalize_processing():
    """Resets the UI state after the batch finishes or is stopped."""
    global is_processing
    is_processing = False
    # Ensure UI updates are run on the main thread
    if root:
        root.after(0, lambda: start_button.config(state=tk.NORMAL))
        root.after(0, lambda: stop_button.config(state=tk.DISABLED))


if __name__ == '__main__':
    # Initialize the GUI
    create_gui()
