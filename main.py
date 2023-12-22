import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess
from  ImageAnalogies import  do_image_analogies
def browse_image(label, image_display):
    file_path = filedialog.askopenfilename()
    label.config(text=file_path)
    if file_path:
        load_and_display_image(file_path, image_display)

def load_and_display_image(file_path, image_display):
    img = Image.open(file_path)
    img.thumbnail((200, 200))  # Adjust the size as needed
    img = ImageTk.PhotoImage(img)
    image_display.config(image=img)
    image_display.image = img

def run_script():
    # Get the paths to the selected images
    image_A_path = label_A.cget("text")
    image_Ap_path = label_Ap.cget("text")
    image_B_path = label_B.cget("text")

    # Collect the other parameters from the user or use defaults
    kappa = entry_kappa.get() if entry_kappa.get() else "0.1"
    n_levels = entry_n_levels.get() if entry_n_levels.get() else "3"
    k_coarse = entry_k_coarse.get() if entry_k_coarse.get() else "5"
    k_fine = entry_k_fine.get() if entry_k_fine.get() else "5"
    output_Bp_path = "output.png"
    # Call the Image Analogies script with the selected image paths and parameters
    do_image_analogies(image_A_path, image_Ap_path, image_B_path,output_Bp_path, Kappa=float(kappa), NLevels=int(n_levels), KCoarse=int(k_coarse), KFine=int(k_fine))


# Create the main window
root = tk.Tk()
root.title("Image Analogies GUI")

# Create a style for a modern look
style = ttk.Style()
style.configure("TButton", padding=5, relief="flat")

# Labels and buttons for image selection
label_A = ttk.Label(root, text="Select Image A:")
label_A.grid(row=0, column=0, padx=10, pady=5, sticky="w")

image_display_A = ttk.Label(root)
image_display_A.grid(row=0, column=2, padx=10, pady=5)

button_A = ttk.Button(root, text="Browse", command=lambda: browse_image(label_A, image_display_A))
button_A.grid(row=0, column=1, padx=10, pady=5)

label_Ap = ttk.Label(root, text="Select Image Ap:")
label_Ap.grid(row=1, column=0, padx=10, pady=5, sticky="w")

image_display_Ap = ttk.Label(root)
image_display_Ap.grid(row=1, column=2, padx=10, pady=5)

button_Ap = ttk.Button(root, text="Browse", command=lambda: browse_image(label_Ap, image_display_Ap))
button_Ap.grid(row=1, column=1, padx=10, pady=5)

label_B = ttk.Label(root, text="Select Image B:")
label_B.grid(row=2, column=0, padx=10, pady=5, sticky="w")

image_display_B = ttk.Label(root)
image_display_B.grid(row=2, column=2, padx=10, pady=5)

button_B = ttk.Button(root, text="Browse", command=lambda: browse_image(label_B, image_display_B))
button_B.grid(row=2, column=1, padx=10, pady=5)

# Entry fields for other parameters with default values
label_kappa = ttk.Label(root, text="Enter Kappa:")
label_kappa.grid(row=3, column=0, padx=10, pady=5, sticky="w")

entry_kappa = ttk.Entry(root)
entry_kappa.insert(0, "0.1")  # Default value
entry_kappa.grid(row=3, column=1, padx=10, pady=5)

label_n_levels = ttk.Label(root, text="Enter NLevels:")
label_n_levels.grid(row=4, column=0, padx=10, pady=5, sticky="w")

entry_n_levels = ttk.Entry(root)
entry_n_levels.insert(0, "3")  # Default value
entry_n_levels.grid(row=4, column=1, padx=10, pady=5)

label_k_coarse = ttk.Label(root, text="Enter KCoarse:")
label_k_coarse.grid(row=5, column=0, padx=10, pady=5, sticky="w")

entry_k_coarse = ttk.Entry(root)
entry_k_coarse.insert(0, "5")  # Default value
entry_k_coarse.grid(row=5, column=1, padx=10, pady=5)

label_k_fine = ttk.Label(root, text="Enter KFine:")
label_k_fine.grid(row=6, column=0, padx=10, pady=5, sticky="w")

entry_k_fine = ttk.Entry(root)
entry_k_fine.insert(0, "5")  # Default value
entry_k_fine.grid(row=6, column=1, padx=10, pady=5)

# Button to run the script
run_button = ttk.Button(root, text="Run Script", command=run_script)
run_button.grid(row=7, columnspan=3, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
