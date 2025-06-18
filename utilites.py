# --- CELL 1: Cloaking Utilities (All functions from cloaking_utils.py) ---
# This cell contains all the necessary functions and initializations for cloaking,
# watermarking, detection, and comparison.

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add
import os
from PIL import Image, ImageTk
import tkinter as tk # Tkinter is primarily for GUI, not directly used in core utilities logic here, but included for completeness if you paste all
import hashlib # For creating a stable watermark from a secret key

# --- Enhanced GAN Generator Definition ---
def build_generator():
    """
    Builds an improved generator model that creates subtle perturbations.
    Uses residual blocks for better quality and U-Net architecture for preserving details.
    """
    inputs = tf.keras.Input(shape=(None, None, 3))

    # Initial preprocessing
    x = Conv2D(64, 7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Downsampling
    x = Conv2D(128, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Residual blocks
    for _ in range(6):
        res = x
        x = Conv2D(256, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, res])

    # Upsampling
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Final output with tanh activation (produces perturbations)
    perturbations = Conv2D(3, 7, padding='same', activation='tanh')(x)

    # Scale down perturbations using a Lambda layer (correct way in Keras)
    # The original scale of 0.05 is good for subtle changes.
    perturbations = tf.keras.layers.Lambda(lambda x: x * 0.05)(perturbations)

    # Add perturbations to original image (identity mapping)
    outputs = Add()([inputs, perturbations])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialize generator once when the module is loaded
generator = build_generator()

# --- Watermarking Configuration ---
WATERMARK_SECRET_KEY = "my_super_secret_cloak_key_123!" # A secret key for the watermark
WATERMARK_STRENGTH = 0.02 # Adjust this for balance between imperceptibility and robustness

def generate_watermark_pattern_dct(size, secret_key):
    """
    Generates a pseudo-random binary pattern for the watermark based on a secret key.
    This pattern is suitable for DCT embedding.
    """
    seed = int(hashlib.sha256(secret_key.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    watermark_pattern = rng.randint(0, 2, size=size).astype(np.float32)
    return watermark_pattern * 2 - 1

def embed_watermark_dct(image_np_rgb, secret_key=WATERMARK_SECRET_KEY, strength=WATERMARK_STRENGTH):
    """
    Embeds an invisible watermark into the DCT coefficients of an image.
    Works on each color channel separately to embed the watermark.
    """
    img_watermarked = image_np_rgb.copy().astype(np.float32)
    h, w, c = img_watermarked.shape
    watermark_embed_size = (h // 8, w // 8)
    watermark_pattern = generate_watermark_pattern_dct(watermark_embed_size, secret_key)

    for i in range(c):
        img_channel = img_watermarked[:, :, i]
        dct_channel = cv2.dct(img_channel)
        embed_row_start = watermark_embed_size[0] // 2 
        embed_col_start = watermark_embed_size[1] // 2 
        block_rows = min(watermark_embed_size[0], h - embed_row_start)
        block_cols = min(watermark_embed_size[1], w - embed_col_start)
        dct_channel[embed_row_start : embed_row_start + block_rows,
                    embed_col_start : embed_col_start + block_cols] += strength * watermark_pattern[:block_rows, :block_cols]
        img_watermarked[:, :, i] = cv2.idct(dct_channel)

    img_watermarked = np.clip(img_watermarked, 0, 255).astype(np.uint8)
    return img_watermarked

def detect_watermark_dct(image_np_rgb, secret_key=WATERMARK_SECRET_KEY, threshold_correlation=0.01):
    """
    Detects the presence of an embedded watermark in the DCT coefficients of an image.
    """
    img = image_np_rgb.copy().astype(np.float32)
    h, w, c = img.shape
    watermark_embed_size = (h // 8, w // 8)
    expected_watermark_pattern = generate_watermark_pattern_dct(watermark_embed_size, secret_key)
    correlations = []

    for i in range(c):
        img_channel = img[:, :, i]
        dct_channel = cv2.dct(img_channel)
        embed_row_start = watermark_embed_size[0] // 2
        embed_col_start = watermark_embed_size[1] // 2
        block_rows = min(watermark_embed_size[0], h - embed_row_start)
        block_cols = min(watermark_embed_size[1], w - embed_col_start)
        extracted_block = dct_channel[embed_row_start : embed_row_start + block_rows,
                                      embed_col_start : embed_col_start + block_cols]

        if extracted_block.shape[0] == 0 or extracted_block.shape[1] == 0 or np.std(extracted_block) < 1e-6:
            correlations.append(0)
            continue

        current_expected_pattern = expected_watermark_pattern[:extracted_block.shape[0], :extracted_block.shape[1]]
        
        if np.std(current_expected_pattern) > 1e-6:
            correlation = np.corrcoef(extracted_block.flatten(), current_expected_pattern.flatten())[0, 1]
            correlations.append(correlation)
        else:
            correlations.append(0)

    avg_correlation = np.mean(correlations)
    print(f"Watermark Detection: Average correlation = {avg_correlation:.4f}, Threshold = {threshold_correlation:.4f}")
    return avg_correlation > threshold_correlation

def cloak_image(image_path, output_dir=None):
    """
    Applies subtle GAN-based perturbations to an image and then embeds a robust watermark.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
    input_tensor = np.expand_dims(img_normalized, axis=0)
    
    try:
        cloaked_img_normalized = generator.predict(input_tensor, verbose=0)[0]
    except Exception as e:
        raise ValueError(f"GAN prediction failed: {str(e)}. Check TensorFlow/GPU setup.")

    cloaked_img_normalized = np.clip(cloaked_img_normalized, -1, 1)
    cloaked_img_pre_watermark = ((cloaked_img_normalized + 1) * 127.5).astype(np.uint8)
    cloaked_img_final = embed_watermark_dct(cloaked_img_pre_watermark)

    cloaked_output_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        cloaked_output_path = os.path.join(output_dir, f"{name}_cloaked_wm.png")
        cloaked_bgr = cv2.cvtColor(cloaked_img_final, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cloaked_output_path, cloaked_bgr)

    return img_rgb, cloaked_img_final, cloaked_output_path

def is_cloaked(image_path, adaptive_threshold=True):
    """
    Determines if an image has been processed by the cloaking utility.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Invalid image file: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        watermark_detected = detect_watermark_dct(img_rgb)
        
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        noise_energy = np.var(laplacian)

        dft = np.fft.fft2(img_gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-10)
        
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        r_outer = min(crow, ccol) * 0.4
        r_inner = min(crow, ccol) * 0.1
        cv2.circle(mask, (ccol, crow), int(r_outer), 1, -1)
        cv2.circle(mask, (ccol, crow), int(r_inner), 0, -1)
        
        high_freq_spectrum = magnitude_spectrum * mask
        high_freq_energy = np.sum(high_freq_spectrum) / (np.sum(mask) + 1e-10)
        
        if adaptive_threshold:
            avg_brightness = np.mean(img_gray)
            threshold = 10 + (avg_brightness / 15.0)
        else:
            threshold = 15.0
        
        cloaked_by_pattern = noise_energy > threshold and high_freq_energy > 70
        
        print(f"Detection Metrics - Noise: {noise_energy:.2f}, Freq: {high_freq_energy:.2f}, Threshold: {threshold:.2f}, Watermark: {watermark_detected}")
        
        return cloaked_by_pattern or watermark_detected

    except Exception as e:
        print(f"Detection error: {str(e)}")
        return False

def compare_images(orig_path, cloaked_path, output_dir=None):
    """
    Generates a visual comparison showing significant differences between two images,
    marked in green.
    """
    orig = cv2.imread(orig_path)
    cloaked = cv2.imread(cloaked_path)

    if orig is None or cloaked is None:
        raise ValueError("Could not read one or both images for comparison.")

    if orig.shape != cloaked.shape:
        cloaked = cv2.resize(cloaked, (orig.shape[1], orig.shape[0]))

    diff = cv2.absdiff(orig, cloaked)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(diff_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    marked = orig.copy()
    marked[thresh == 255] = [0, 255, 0]

    comparison_output_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(orig_path)
        name, ext = os.path.splitext(base_name)
        comparison_output_path = os.path.join(output_dir, f"{name}_comparison{ext}")
        cv2.imwrite(comparison_output_path, marked)
        
    return cv2.cvtColor(marked, cv2.COLOR_BGR2RGB), comparison_output_path

def show_image_on_canvas(canvas, image_path, max_width=600, max_height=400):
    """
    Loads an image from path, resizes it to fit within max_width/height,
    and displays it on the given Tkinter canvas.
    """
    try:
        img_pil = Image.open(image_path)
        img_pil.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_pil)
        canvas.delete("all")
        canvas.config(width=photo.width(), height=photo.height())
        canvas.create_image(photo.width() / 2, photo.height() / 2, image=photo)
        return photo
    except Exception as e:
        print(f"Error displaying image on canvas: {e}")
        return None

def show_numpy_image_on_canvas(canvas, img_np_rgb, max_width=600, max_height=400):
    """
    Displays a NumPy array image (RGB) on the given Tkinter canvas.
    """
    try:
        img_pil = Image.fromarray(img_np_rgb)
        img_pil.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_pil)
        canvas.delete("all")
        canvas.config(width=photo.width(), height=photo.height())
        canvas.create_image(photo.width() / 2, photo.height() / 2, image=photo)
        return photo
    except Exception as e:
        print(f"Error displaying numpy image on canvas: {e}")
        return None


# --- CELL 2: Tkinter Application (from your cloak-image-app-fixed-v2) ---
# This cell contains your GUI application.

class CloakImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cloak Image - AI Training Disruptor")
        self.root.geometry("700x550")
        self.root.resizable(False, False)
        
        self.output_dir = os.path.join(os.path.expanduser("~"), "CloakedImages")
        self.original_image_path = ""
        self.cloaked_image_path = ""
        self.original_photo = None
        self.cloaked_photo = None
        
        self.create_widgets()
        os.makedirs(self.output_dir, exist_ok=True)

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack(expand=True, fill=tk.BOTH)

        tk.Label(main_frame, text="AI Image Cloaker", font=("Arial", 16, "bold")).pack(pady=10)
        tk.Label(main_frame, text=f"Select an image to apply AI-disrupting cloaking.\nCloaked images saved to: {self.output_dir}", 
                 wraplength=600).pack(pady=5)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Select Image", command=self.select_image_for_cloaking,
                  font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", relief=tk.RAISED, bd=3).pack(side=tk.LEFT, padx=10)
        
        self.cloak_button = tk.Button(button_frame, text="Cloak & Save", command=self.perform_cloaking,
                                       font=("Arial", 10, "bold"), bg="#2196F3", fg="white", relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.cloak_button.pack(side=tk.LEFT, padx=10)

        display_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        display_frame.pack(expand=True, fill=tk.BOTH, pady=10)

        original_frame = tk.Frame(display_frame)
        original_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        tk.Label(original_frame, text="Original Image", font=("Arial", 10, "italic")).pack()
        self.original_canvas = tk.Canvas(original_frame, bg="lightgray", width=300, height=300, relief=tk.SUNKEN, bd=1)
        self.original_canvas.pack(expand=True, fill=tk.BOTH)

        cloaked_frame = tk.Frame(display_frame)
        cloaked_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        tk.Label(cloaked_frame, text="Cloaked Image", font=("Arial", 10, "italic")).pack()
        self.cloaked_canvas = tk.Canvas(cloaked_frame, bg="lightgray", width=300, height=300, relief=tk.SUNKEN, bd=1)
        self.cloaked_canvas.pack(expand=True, fill=tk.BOTH)

        self.status_label = tk.Label(main_frame, text="Ready: Select an image to begin.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def select_image_for_cloaking(self):
        file_types = [
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG Files", "*.jpg *.jpeg"),
            ("PNG Files", "*.png"),
            ("All Files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(title="Select Image to Cloak", filetypes=file_types)
        
        if file_path:
            self.original_image_path = file_path
            self.cloaked_image_path = ""
            self.update_status(f"Selected: {os.path.basename(file_path)}")
            
            self.original_canvas.delete("all")
            self.cloaked_canvas.delete("all")
            
            try:
                img = Image.open(file_path)
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                self.original_photo = ImageTk.PhotoImage(img)
                self.original_canvas.create_image(150, 150, anchor=tk.CENTER, image=self.original_photo)
                self.cloak_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.update_status("Error loading image.")
                self.cloak_button.config(state=tk.DISABLED)
        else:
            self.update_status("Image selection cancelled.")

    def perform_cloaking(self):
        if not self.original_image_path:
            messagebox.showwarning("No Image Selected", "Please select an image first.")
            return

        try:
            self.update_status("Processing: Applying cloaking perturbations...")
            self.cloak_button.config(state=tk.DISABLED)
            self.root.update_idletasks()

            # Call the cloak_image function from the utility definitions above
            _, cloaked_img_np_rgb, cloaked_path = cloak_image(self.original_image_path, self.output_dir)

            if cloaked_path and os.path.exists(cloaked_path):
                self.cloaked_image_path = cloaked_path
                img = Image.open(cloaked_path)
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                self.cloaked_photo = ImageTk.PhotoImage(img)
                self.cloaked_canvas.create_image(150, 150, anchor=tk.CENTER, image=self.cloaked_photo)
                
                self.update_status(f"Cloaking complete! Saved to: {os.path.basename(cloaked_path)}")
                messagebox.showinfo("Success", f"Image cloaked successfully!\nSaved to:\n{cloaked_path}")
            else:
                messagebox.showerror("Error", "Cloaking failed to save the image. Check console for details if any.")
                self.update_status("Cloaking failed. See console for errors.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during cloaking: {str(e)}\n\n"
                                          "Please check your Python console/terminal for a detailed traceback.")
            self.update_status(f"Error: {str(e)}. Check console.")
        finally:
            self.cloak_button.config(state=tk.NORMAL)

# --- CELL 3: AI Testing Script ---
# This cell contains the script to evaluate cloaking effectiveness against a pre-trained AI.

from tensorflow.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
# Note: cloak_image and is_cloaked are defined in CELL 1

def run_ai_disruption_test():
    """
    Runs a test to evaluate how effectively cloaked images disrupt a pre-trained AI model.
    """
    TEST_IMAGES_DIR = 'test_images' # Ensure this folder exists and has images
    CLOAKED_OUTPUT_DIR = 'test_output/cloaked'
    COMPARISON_OUTPUT_DIR = 'test_output/comparison'

    os.makedirs(CLOAKED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)

    print("Loading pre-trained ResNet50 model (this may take a moment)...")
    try:
        model = ResNet50(weights='imagenet')
        print("ResNet50 model loaded successfully.")
    except Exception as e:
        print(f"Error loading ResNet50 model: {e}")
        print("Please ensure you have an active internet connection or try installing TensorFlow/Keras dependencies properly.")
        return # Exit the function if model can't be loaded

    print("\n--- Starting AI Disruption Test ---")
    print(f"Testing images from: {TEST_IMAGES_DIR}")
    print(f"Cloaked outputs saved to: {CLOAKED_OUTPUT_DIR}")

    total_images_tested = 0
    successfully_disrupted = 0
    watermark_detected_on_cloaked = 0

    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print(f"No image files found in {TEST_IMAGES_DIR}. Please add some images to test.")
        return

    for filename in image_files:
        original_image_path = os.path.join(TEST_IMAGES_DIR, filename)
        print(f"\n--- Processing image: {filename} ---")

        total_images_tested += 1

        # --- Step 1: Cloak the image using your utility ---
        try:
            original_img_np, cloaked_img_np, saved_cloaked_path = cloak_image(original_image_path, CLOAKED_OUTPUT_DIR)
            if not saved_cloaked_path:
                print(f"Skipping {filename}: Cloaking failed to save output.")
                continue
        except Exception as e:
            print(f"Error cloaking {filename}: {e}. Skipping this image.")
            continue

        # --- Step 2: Prepare images for ResNet50 model ---
        img_original_resized = cv2.resize(original_img_np, (224, 224))
        img_cloaked_resized = cv2.resize(cloaked_img_np, (224, 224))

        input_original = preprocess_input(np.expand_dims(img_original_resized, axis=0))
        input_cloaked = preprocess_input(np.expand_dims(img_cloaked_resized, axis=0))

        # --- Step 3: Get predictions from the pre-trained AI model ---
        print("Getting AI predictions...")
        preds_original = model.predict(input_original, verbose=0)
        preds_cloaked = model.predict(input_cloaked, verbose=0)

        top_original_label = decode_predictions(preds_original, top=1)[0][0][1]
        top_cloaked_label = decode_predictions(preds_cloaked, top=1)[0][0][1]

        print(f"Original AI prediction: {top_original_label}")
        print(f"Cloaked AI prediction:  {top_cloaked_label}")

        # --- Step 4: Evaluate AI Disruption ---
        if top_original_label != top_cloaked_label:
            print(f"RESULT: SUCCESS! Cloaking caused AI to misclassify from '{top_original_label}' to '{top_cloaked_label}'.")
            successfully_disrupted += 1
        else:
            print(f"RESULT: Cloaking DID NOT disrupt AI. Still classified as '{top_original_label}'.")

        # --- Step 5: Check your 'is_cloaked' utility on the cloaked image ---
        try:
            is_it_cloaked_by_util = is_cloaked(saved_cloaked_path)
            print(f"Your 'is_cloaked' utility says it's cloaked: {is_it_cloaked_by_util}")
            if is_it_cloaked_by_util:
                watermark_detected_on_cloaked += 1
        except Exception as e:
            print(f"Error calling is_cloaked utility: {e}")
        
        # --- Step 6: (Optional) Test the screenshot loophole using is_cloaked ---
        screenshot_test_path = os.path.join(CLOAKED_OUTPUT_DIR, f"screenshot_of_{filename.split('.')[0]}_cloaked_wm.png")
        if os.path.exists(screenshot_test_path):
            print(f"--- Testing 'is_cloaked' on a simulated screenshot ({os.path.basename(screenshot_test_path)}) ---")
            is_screenshot_cloaked_by_util = is_cloaked(screenshot_test_path)
            print(f"Your 'is_cloaked' utility says screenshot is cloaked: {is_screenshot_cloaked_by_util}")
        else:
            print(f"To test screenshot robustness, manually take a screenshot of '{os.path.basename(saved_cloaked_path)}'")
            print(f"and save it as '{os.path.basename(screenshot_test_path)}' then re-run the script.")


    # --- Test Summary ---
    print("\n--- AI Disruption Test Summary ---")
    print(f"Total images tested: {total_images_tested}")
    print(f"Images where AI classification was disrupted: {successfully_disrupted}")
    if total_images_tested > 0:
        disruption_rate = (successfully_disrupted / total_images_tested) * 100
        print(f"AI Disruption Rate (ASR): {disruption_rate:.2f}%")
        print(f"Your 'is_cloaked' utility detected cloak/watermark on {watermark_detected_on_cloaked} out of {total_images_tested} directly cloaked images.")
    else:
        print("No images were successfully processed for testing.")


# --- CELL 4: Execution Blocks (Run these separately based on what you want to do) ---

# To run the Tkinter GUI:
# if __name__ == "__main__": # In Jupyter, this check often behaves differently, so directly call if needed.
#     root = tk.Tk()
#     app = CloakImageApp(root)
#     root.mainloop()

# To run the AI Disruption Test (after creating 'test_images' folder with images):
# run_ai_disruption_test()
