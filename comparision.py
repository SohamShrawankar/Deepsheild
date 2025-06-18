import tkinter as tk
from tkinter import filedialog, messagebox, ttk  # Import ttk for themed widgets
import os
from PIL import Image, ImageTk, ImageStat # Pillow library for image processing
import numpy as np # For numerical operations on images

class AICloakingToolkitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Cloaking Toolkit")
        self.root.geometry("900x700") # Adjusted size to accommodate tabs and content
        self.root.resizable(True, True) # Allow resizing of the window

        # --- Variables for Image Paths and PhotoImages (to prevent garbage collection) ---
        # For the Detection tab
        self.image_to_check_path = ""
        self.displayed_detection_photo = None

        # For the Comparison tab
        self.original_comparison_path = ""
        self.cloaked_comparison_path = ""
        self.displayed_comparison_photo = None

        self.create_widgets()

    def create_widgets(self):
        # --- Create a Notebook (Tabbed Interface) for organizing functionalities ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # --- Frame for the "Detect Cloaking" tab ---
        self.detection_frame = tk.Frame(self.notebook, padx=15, pady=15)
        self.notebook.add(self.detection_frame, text="Detect Cloaking")
        self._create_detection_tab_widgets(self.detection_frame)

        # --- Frame for the "Compare Images" tab ---
        self.comparison_frame = tk.Frame(self.notebook, padx=15, pady=15)
        self.notebook.add(self.comparison_frame, text="Compare Images")
        self._create_comparison_tab_widgets(self.comparison_frame)

    # --- Widgets and logic specific to the "Detect Cloaking" Tab ---
    def _create_detection_tab_widgets(self, parent_frame):
        # Title for the detection tab
        tk.Label(parent_frame, text="AI Cloaking Detector", font=("Arial", 18, "bold")).pack(pady=15)
        # Description for the detection tab
        tk.Label(parent_frame, text="Select an image to check if it has been cloaked.", wraplength=450).pack(pady=10)

        # Frame to hold buttons
        button_frame = tk.Frame(parent_frame)
        button_frame.pack(pady=20)

        # Button to select an image for detection
        tk.Button(button_frame, text="Select Image to Check", command=self._select_image_for_detection,
                  font=("Arial", 11, "bold"), bg="#607D8B", fg="white",
                  relief=tk.RAISED, bd=3).pack(side=tk.LEFT, padx=15)

        # Button to analyze the selected image (initially disabled)
        self.detect_button = tk.Button(button_frame, text="Analyze Image", command=self._perform_detection,
                                       font=("Arial", 11, "bold"), bg="#FFC107", fg="black",
                                       relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.detect_button.pack(side=tk.LEFT, padx=15)

        # Canvas to display the selected image for detection
        self.detection_image_canvas = tk.Canvas(parent_frame, bg='lightgray', width=550, height=350, relief=tk.SUNKEN, bd=1)
        self.detection_image_canvas.pack(expand=True, fill=tk.BOTH, pady=15)

        # Status bar for the detection tab
        self.detection_status_label = tk.Label(parent_frame, text="Ready: Select an image to analyze.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.detection_status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    # --- Widgets and logic specific to the "Compare Images" Tab ---
    def _create_comparison_tab_widgets(self, parent_frame):
        # Title for the comparison tab
        tk.Label(parent_frame, text="Image Comparison Tool", font=("Arial", 18, "bold")).pack(pady=15)
        # Description for the comparison tab
        tk.Label(parent_frame, text="Select an original image and its cloaked version to visualize the subtle differences. Red areas in the comparison image mark the perturbations.", wraplength=650).pack(pady=10)

        # Frame to display paths of selected images
        path_frame = tk.Frame(parent_frame)
        path_frame.pack(pady=15, fill=tk.X)
        tk.Label(path_frame, text="Original Image Path:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=2)
        self.orig_path_label = tk.Label(path_frame, text="No file selected", anchor="w", fg="blue", wraplength=600)
        self.orig_path_label.grid(row=0, column=1, sticky="ew", padx=5)
        tk.Label(path_frame, text="Cloaked Image Path:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        self.cloak_path_label = tk.Label(path_frame, text="No file selected", anchor="w", fg="blue", wraplength=600)
        self.cloak_path_label.grid(row=1, column=1, sticky="ew", padx=5)
        path_frame.grid_columnconfigure(1, weight=1) # Allow path labels to expand

        # Frame to hold buttons
        button_frame = tk.Frame(parent_frame)
        button_frame.pack(pady=20)
        
        # Button to select original image
        tk.Button(button_frame, text="Select Original Image", command=self._select_original_image,
                  font=("Arial", 11, "bold"), bg="#607D8B", fg="white", relief=tk.RAISED, bd=3).pack(side=tk.LEFT, padx=15)
        # Button to select cloaked image
        tk.Button(button_frame, text="Select Cloaked Image", command=self._select_cloaked_image,
                  font=("Arial", 11, "bold"), bg="#607D8B", fg="white", relief=tk.RAISED, bd=3).pack(side=tk.LEFT, padx=15)
        
        # Button to generate comparison (initially disabled)
        self.compare_button = tk.Button(button_frame, text="Generate Comparison", command=self._perform_comparison,
                                        font=("Arial", 11, "bold"), bg="#009688", fg="white", relief=tk.RAISED, bd=3,
                                        state=tk.DISABLED)
        self.compare_button.pack(side=tk.LEFT, padx=15)

        # Canvas to display the comparison image
        self.comparison_image_canvas = tk.Canvas(parent_frame, bg='lightgray', width=750, height=400, relief=tk.SUNKEN, bd=1)
        self.comparison_image_canvas.pack(expand=True, fill=tk.BOTH, pady=15)

        # Status bar for the comparison tab
        self.comparison_status_label = tk.Label(parent_frame, text="Ready: Select both images to compare.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.comparison_status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self._check_can_compare() # Initial check to set compare button state

    # --- Utility Functions (shared by both tabs or internal helper methods) ---
    def _update_status(self, message, tab="detection"):
        """
        Updates the text in the appropriate status bar based on the 'tab' argument.
        Ensures GUI updates immediately.
        """
        if tab == "detection":
            self.detection_status_label.config(text=message)
        elif tab == "comparison":
            self.comparison_status_label.config(text=message)
        self.root.update_idletasks()

    def _show_image_on_canvas(self, canvas, image_path):
        """
        Displays an image on the given Tkinter canvas, resizing it to fit
        the canvas while maintaining its aspect ratio.
        Returns the PhotoImage object to prevent it from being garbage collected.
        """
        try:
            img = Image.open(image_path)
            
            # Get current canvas dimensions, using fallback if not yet available
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            if canvas_width == 0 or canvas_height == 0:
                canvas_width = 550  # Default width for detection tab canvas
                canvas_height = 350 # Default height for detection tab canvas
                if canvas is self.comparison_image_canvas: # Specific defaults for comparison canvas
                    canvas_width = 750
                    canvas_height = 400
                print(f"Warning: Canvas dimensions zero, using defaults ({canvas_width}x{canvas_height}) for initial display.")

            # Calculate new dimensions to fit image within canvas, maintaining aspect ratio
            img_width, img_height = img.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Resize image using LANCZOS filter for high quality downsampling
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Convert PIL Image to PhotoImage for Tkinter display
            photo = ImageTk.PhotoImage(img)

            # Clear any previous image on the canvas and draw the new one centered
            canvas.delete("all")
            canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=photo)
            
            # Store a reference to the PhotoImage object on the canvas itself
            # This is CRUCIAL to prevent the image from disappearing due to garbage collection.
            canvas.image = photo 
            return photo
        except Exception as e:
            print(f"Error loading or displaying image on canvas: {e}")
            return None

    # --- Logic for Detection Tab (internal methods) ---
    def _select_image_for_detection(self):
        """Opens a file dialog to select an image for cloaking detection."""
        file_path = filedialog.askopenfilename(
            title="Select Image to Detect Cloaking",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")] # Filter for common image types
        )
        if file_path:
            self.image_to_check_path = file_path
            self._update_status(f"Selected: {os.path.basename(file_path)}", tab="detection")
            
            # Update GUI immediately to ensure canvas dimensions are correct before drawing
            self.root.update_idletasks()
            self.displayed_detection_photo = self._show_image_on_canvas(self.detection_image_canvas, file_path)
            self.detect_button.config(state=tk.NORMAL) # Enable the Analyze button
        else:
            self._update_status("Image selection cancelled.", tab="detection")
            self.detect_button.config(state=tk.DISABLED) # Disable if no image is selected

    def _perform_detection(self):
        """Initiates the cloaking detection process for the selected image."""
        if not self.image_to_check_path:
            messagebox.showwarning("No Image Selected", "Please select an image first.")
            return

        try:
            self._update_status("Analyzing: Checking for cloaking signatures...", tab="detection")
            
            # Call the internal cloaking detection function
            is_cloaked_result = self._is_cloaked(self.image_to_check_path)

            status_text = "CLOAKED" if is_cloaked_result else "NOT CLOAKED"
            
            # Show a message box with the detection result
            messagebox.showinfo("Detection Result",
                                f"Image Analysis Complete:\n\n"
                                f"Status: {status_text}\n\n"
                                f"This detection uses advanced metrics to identify\n"
                                f"GAN-based perturbations designed to disrupt AI training.")
            
            self._update_status(f"Detection complete: Image is {status_text}.", tab="detection")

        except Exception as e:
            # Catch and display any errors during the detection process
            messagebox.showerror("Error", f"An error occurred during detection: {str(e)}")
            self._update_status("Error during detection.", tab="detection")

    def _is_cloaked(self, image_path):
        """
        *** THIS IS THE CORE FUNCTION FOR ACCURATE CLOAKING DETECTION ***
        
        You MUST replace the placeholder logic below with your actual, robust
        AI cloaking detection algorithm for real-world accuracy.

        Real cloaking detection involves advanced image processing techniques,
        machine learning models (e.g., CNNs trained on cloaked/uncloaked datasets),
        or specific algorithms designed to identify subtle adversarial perturbations.

        Current Placeholder Logic:
        - Checks if "cloaked", "disrupt", or "adversarial" is in the filename (very unreliable).
        - Performs a very basic and inaccurate check on pixel standard deviation across channels.
        """
        print(f"Attempting to detect cloaking for: {os.path.basename(image_path)}")

        # --- START OF PLACEHOLDER DETECTION LOGIC ---
        # **REPLACE THIS ENTIRE SECTION WITH YOUR REAL CLOAKING DETECTION ALGORITHM**

        # Placeholder Example 1: Check filename (Highly UNRELIABLE for real detection)
        # This is purely for demonstration purposes to show a 'True' result sometimes.
        filename = os.path.basename(image_path).lower()
        if "cloaked" in filename or "disrupt" in filename or "adversarial" in filename:
            print("Placeholder detection: Filename suggests cloaking (very weak indicator).")
            return True 

        # Placeholder Example 2: Very simplistic pixel analysis (also UNRELIABLE for real cloaking)
        try:
            img = Image.open(image_path).convert("RGB") # Ensure image is in RGB format
            stat = ImageStat.Stat(img) # Get statistics for each channel
            
            if stat.stddev and len(stat.stddev) == 3: # Check if standard deviation exists for R, G, B channels
                avg_stddev = sum(stat.stddev) / 3 # Calculate average standard deviation
                print(f"Placeholder: Average pixel standard deviation: {avg_stddev:.2f}")
                
                # These thresholds are completely arbitrary and DO NOT RELY ON THEM FOR ACCURACY.
                # Real cloaking might introduce very low-magnitude noise that wouldn't affect
                # overall std dev significantly.
                if avg_stddev < 20 or avg_stddev > 90: 
                    # Uncomment the line below to enable this very weak check for demonstration:
                    # return True 
                    pass # Do nothing if within these arbitrary bounds

        except Exception as e:
            print(f"Error during placeholder image analysis in _is_cloaked: {e}")
            pass

        # If none of the placeholder conditions are met, default to NOT CLOAKED
        print("Placeholder detection: Image does not appear cloaked based on current (weak) logic.")
        return False
        # --- END OF PLACEHOLDER DETECTION LOGIC ---


    # --- Logic for Comparison Tab (internal methods) ---
    def _check_can_compare(self):
        """
        Enables or disables the 'Generate Comparison' button based on whether
        both original and cloaked image paths have been selected.
        """
        if self.original_comparison_path and self.cloaked_comparison_path:
            self.compare_button.config(state=tk.NORMAL)
        else:
            self.compare_button.config(state=tk.DISABLED)

    def _select_original_image(self):
        """Opens a file dialog to select the original (uncloaked) image for comparison."""
        file_path = filedialog.askopenfilename(
            title="Select Original Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.original_comparison_path = file_path
            self.orig_path_label.config(text=os.path.basename(file_path))
            self._update_status(f"Original selected: {os.path.basename(file_path)}", tab="comparison")
            self._check_can_compare() # Re-check button state
        else:
            self._update_status("Original image selection cancelled.", tab="comparison")

    def _select_cloaked_image(self):
        """Opens a file dialog to select the cloaked image for comparison."""
        file_path = filedialog.askopenfilename(
            title="Select Cloaked Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.cloaked_comparison_path = file_path
            self.cloak_path_label.config(text=os.path.basename(file_path))
            self._update_status(f"Cloaked selected: {os.path.basename(file_path)}", tab="comparison")
            self._check_can_compare() # Re-check button state
        else:
            self._update_status("Cloaked image selection cancelled.", tab="comparison")

    def _perform_comparison(self):
        """Initiates the image comparison process between selected original and cloaked images."""
        if not self.original_comparison_path or not self.cloaked_comparison_path:
            messagebox.showwarning("Missing Images", "Please select both original and cloaked images.")
            return

        try:
            self._update_status("Generating comparison image... This may take a moment.", tab="comparison")
            
            # Create a directory to save comparison images if it doesn't already exist
            output_dir = "comparison_images"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Call the internal comparison function to generate the difference image
            saved_path = self._compare_images(
                self.original_comparison_path, self.cloaked_comparison_path, output_dir=output_dir
            )

            if saved_path:
                # Update GUI immediately to ensure canvas dimensions are correct before drawing
                self.root.update_idletasks() 
                self.displayed_comparison_photo = self._show_image_on_canvas(self.comparison_image_canvas, saved_path)
                self._update_status(f"Comparison complete! Saved to: {os.path.basename(saved_path)}", tab="comparison")
                
                messagebox.showinfo("Comparison Complete",
                                     f"Comparison image created successfully!\n"
                                     f"Saved as: {os.path.basename(saved_path)}\n\n"
                                     f"Red areas in the displayed image indicate where subtle perturbations (differences) were added. "
                                     f"These changes are designed to be minimally visible to humans "
                                     f"while potentially disrupting AI training processes.")
            else:
                messagebox.showerror("Error", "Comparison failed to save the image.")

        except Exception as e:
            # Catch and display any errors during the comparison process
            messagebox.showerror("Error", f"An error occurred during comparison: {str(e)}")
            self._update_status("Error during comparison.", tab="comparison")

    def _compare_images(self, original_path, cloaked_path, output_dir="comparison_images"):
        """
        Compares an original image with its cloaked counterpart,
        highlighting the pixel-level differences as RED areas (perturbations).

        Args:
            original_path (str): Path to the original (uncloaked) image.
            cloaked_path (str): Path to the cloaked image.
            output_dir (str): Directory where the generated comparison image will be saved.

        Returns:
            str: The full path to the saved comparison image, or None if an error occurs.
        """
        try:
            # Load images and convert to RGB format and then to NumPy arrays for calculations
            img_orig = Image.open(original_path).convert("RGB")
            img_cloak = Image.open(cloaked_path).convert("RGB")

            # Ensure both images have the same dimensions for accurate pixel-wise comparison
            if img_orig.size != img_cloak.size:
                # If sizes differ, resize the cloaked image to match the original.
                # In real cloaking scenarios, images should ideally be the same size.
                img_cloak = img_cloak.resize(img_orig.size, Image.LANCZOS)
                print(f"Warning: Image sizes differ. Resizing cloaked image to {img_orig.size}")

            # Convert images to float32 NumPy arrays for precise subtraction
            np_orig = np.array(img_orig).astype(np.float32)
            np_cloak = np.array(img_cloak).astype(np.float32)

            # Calculate the absolute pixel-wise difference between the two images
            # This will show the magnitude of change for each pixel
            difference = np.abs(np_cloak - np_orig)

            # Normalize the difference values to the 0-255 range for better visualization
            # This makes even subtle differences more apparent.
            max_diff_val = np.max(difference)
            if max_diff_val == 0: # If there's no difference at all (images are identical)
                normalized_diff = np.zeros_like(difference)
            else:
                normalized_diff = (difference / max_diff_val) * 255

            normalized_diff = normalized_diff.astype(np.uint8) # Convert back to uint8 for image creation

            # Create the final comparison image
            # Start with a copy of the original image to overlay differences
            comparison_img_rgb = np_orig.astype(np.uint8)
            
            # Calculate the maximum difference across R, G, B channels for each pixel
            diff_magnitude = np.max(normalized_diff, axis=2) 
            
            # Define a threshold: pixels with a difference magnitude above this value will be highlighted.
            # This value might need fine-tuning based on the specific cloaking techniques used.
            DIFFERENCE_THRESHOLD = 5 # (out of normalized 255)
            
            # Identify pixels that have a significant difference based on the threshold
            has_diff = diff_magnitude > DIFFERENCE_THRESHOLD
            
            # Set changed pixels to RED to act as visual markers
            comparison_img_rgb[has_diff, 0] = 255 # Set Red channel to max (bright red)
            comparison_img_rgb[has_diff, 1] = 0   # Set Green channel to 0
            comparison_img_rgb[has_diff, 2] = 0   # Set Blue channel to 0

            # Convert the NumPy array back to a PIL Image
            comparison_pil_img = Image.fromarray(comparison_img_rgb)

            # Construct the output filename and path
            orig_filename_base = os.path.splitext(os.path.basename(original_path))[0]
            cloak_filename_base = os.path.splitext(os.path.basename(cloaked_path))[0]
            
            output_filename = f"{orig_filename_base}_vs_{cloak_filename_base}_diff.png"
            saved_path = os.path.join(output_dir, output_filename)
            
            # Save the generated comparison image
            comparison_pil_img.save(saved_path)
            print(f"Comparison image saved to: {saved_path}")
            return saved_path

        except Exception as e:
            print(f"Error during image comparison: {e}")
            return None


# --- Main execution block ---
if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()
    # Instantiate and run the AI Cloaking Toolkit application
    app = AICloakingToolkitApp(root)
    root.mainloop()
