import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

class CloakImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cloak Image - AI Training Disruptor")
        self.root.geometry("800x600")  # Increased size for watermark controls
        self.root.resizable(False, False)

        # Watermark settings
        self.watermark_text = "CONFIDENTIAL"
        self.watermark_font = "arial.ttf"
        self.watermark_size = 20
        self.watermark_color = (255, 255, 255, 128)  # White with transparency
        self.watermark_position = "bottom-right"  # Default position

        # Set the custom output directory
        self.output_dir = r"C:\Users\Asus\OneDrive\Pictures\Cloaking"
        
        self.original_image_path = ""
        self.cloaked_image_path = ""
        self.original_photo = None
        self.cloaked_photo = None

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack(expand=True, fill=tk.BOTH)

        tk.Label(main_frame, text="AI Image Cloaker", font=("Arial", 16, "bold")).pack(pady=10)
        tk.Label(main_frame, 
                text=f"Select an image to apply AI-disrupting cloaking. Cloaked images will be saved to:\n{self.output_dir}", 
                wraplength=600).pack(pady=5)

        # Watermark controls frame
        watermark_frame = tk.Frame(main_frame)
        watermark_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(watermark_frame, text="Set Watermark Text", command=self.set_watermark_text,
                 font=("Arial", 9), bg="#E0E0E0").pack(side=tk.LEFT, padx=5)
        
        tk.Button(watermark_frame, text="Set Watermark Size", command=self.set_watermark_size,
                 font=("Arial", 9), bg="#E0E0E0").pack(side=tk.LEFT, padx=5)
        
        tk.Button(watermark_frame, text="Set Watermark Color", command=self.set_watermark_color,
                 font=("Arial", 9), bg="#E0E0E0").pack(side=tk.LEFT, padx=5)
        
        tk.Button(watermark_frame, text="Set Position", command=self.set_watermark_position,
                 font=("Arial", 9), bg="#E0E0E0").pack(side=tk.LEFT, padx=5)

        # Buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Select Image", command=self.select_image_for_cloaking,
                 font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
                 relief=tk.RAISED, bd=3).pack(side=tk.LEFT, padx=10)
        
        self.cloak_button = tk.Button(button_frame, text="Cloak & Save", command=self.perform_cloaking,
                                     font=("Arial", 10, "bold"), bg="#2196F3", fg="white",
                                     relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.cloak_button.pack(side=tk.LEFT, padx=10)

        # Image display areas
        display_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        display_frame.pack(expand=True, fill=tk.BOTH, pady=10)

        # Original Image
        original_frame = tk.Frame(display_frame)
        original_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        tk.Label(original_frame, text="Original Image", font=("Arial", 10, "italic")).pack()
        self.original_canvas = tk.Canvas(original_frame, bg='lightgray', width=350, height=350, relief=tk.SUNKEN, bd=1)
        self.original_canvas.pack(expand=True, fill=tk.BOTH)

        # Cloaked Image
        cloaked_frame = tk.Frame(display_frame)
        cloaked_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        tk.Label(cloaked_frame, text="Cloaked Image", font=("Arial", 10, "italic")).pack()
        self.cloaked_canvas = tk.Canvas(cloaked_frame, bg='lightgray', width=350, height=350, relief=tk.SUNKEN, bd=1)
        self.cloaked_canvas.pack(expand=True, fill=tk.BOTH)

        # Status Label
        self.status_label = tk.Label(main_frame, text="Ready: Select an image to begin.", 
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def set_watermark_text(self):
        new_text = simpledialog.askstring("Watermark Text", "Enter watermark text:", 
                                         initialvalue=self.watermark_text)
        if new_text:
            self.watermark_text = new_text
            self.update_status(f"Watermark text set to: {self.watermark_text}")
            if self.cloaked_image_path:  # Refresh display if we have a cloaked image
                self.display_cloaked_image(self.cloaked_image_path)

    def set_watermark_size(self):
        new_size = simpledialog.askinteger("Watermark Size", "Enter font size (8-72):", 
                                          initialvalue=self.watermark_size,
                                          minvalue=8, maxvalue=72)
        if new_size:
            self.watermark_size = new_size
            self.update_status(f"Watermark size set to: {self.watermark_size}")
            if self.cloaked_image_path:
                self.display_cloaked_image(self.cloaked_image_path)

    def set_watermark_color(self):
        color = tk.colorchooser.askcolor(title="Choose Watermark Color", 
                                        initialcolor=self.watermark_color[:3])[0]
        if color:
            self.watermark_color = (int(color[0]), int(color[1]), int(color[2]), 128)
            self.update_status(f"Watermark color set to: {color}")
            if self.cloaked_image_path:
                self.display_cloaked_image(self.cloaked_image_path)
    
    def set_watermark_position(self):
        position = simpledialog.askstring("Watermark Position", 
                                         "Enter position (top-left, top-right, bottom-left, bottom-right):",
                                         initialvalue=self.watermark_position)
        if position and position.lower() in ["top-left", "top-right", "bottom-left", "bottom-right"]:
            self.watermark_position = position.lower()
            self.update_status(f"Watermark position set to: {self.watermark_position}")
            if self.cloaked_image_path:
                self.display_cloaked_image(self.cloaked_image_path)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def select_image_for_cloaking(self):
        file_path = filedialog.askopenfilename(
            title="Select Image to Cloak",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.original_image_path = file_path
            self.cloaked_image_path = ""
            self.update_status(f"Selected: {os.path.basename(file_path)}")
            
            self.original_canvas.delete("all")
            self.cloaked_canvas.delete("all")
            
            try:
                img = Image.open(file_path)
                img.thumbnail((350, 350))
                self.original_photo = ImageTk.PhotoImage(img)
                self.original_canvas.create_image(
                    175, 175, anchor=tk.CENTER, image=self.original_photo
                )
                self.cloak_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.update_status("Error loading image.")
        else:
            self.update_status("Image selection cancelled.")

    def add_watermark(self, image):
        """Adds watermark to the image in the specified position"""
        try:
            # Make sure the image is in RGBA mode
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Create a transparent layer for the watermark
            watermark = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark)
            
            # Use a default font if the specified one isn't available
            try:
                font = ImageFont.truetype(self.watermark_font, self.watermark_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size
            text_bbox = draw.textbbox((0, 0), self.watermark_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            margin = 20  # 20px margin from edges
            
            # Calculate position based on selected option
            if self.watermark_position == "bottom-right":
                x = image.width - text_width - margin
                y = image.height - text_height - margin
            elif self.watermark_position == "top-right":
                x = image.width - text_width - margin
                y = margin
            elif self.watermark_position == "bottom-left":
                x = margin
                y = image.height - text_height - margin
            else:  # top-left
                x = margin
                y = margin
            
            # Draw the text
            draw.text((x, y), self.watermark_text, font=font, fill=self.watermark_color)
            
            # Combine with original image
            watermarked = Image.alpha_composite(image, watermark)
            
            return watermarked
            
        except Exception as e:
            print(f"Error adding watermark: {e}")
            return image  # Return original if watermark fails

    def display_cloaked_image(self, image_path):
        """Display the cloaked image with watermark on canvas"""
        try:
            img = Image.open(image_path)
            img.thumbnail((350, 350))
            self.cloaked_photo = ImageTk.PhotoImage(img)
            self.cloaked_canvas.create_image(
                175, 175, anchor=tk.CENTER, image=self.cloaked_photo
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display cloaked image: {str(e)}")

    def cloak_image(self, image_path):
        """Apply cloaking perturbations and watermark to an image"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Load the image
            img = Image.open(image_path)
            
            # Convert to numpy array for processing
            img_array = np.array(img)
            
            # Convert to float for calculations
            img_float = img_array.astype(np.float32) / 255.0
            
            # Generate random perturbations
            perturbation = np.random.normal(loc=0, scale=0.05, size=img_float.shape)
            
            # Apply perturbations
            cloaked_img = img_float + perturbation
            cloaked_img = np.clip(cloaked_img, 0, 1)  # Ensure valid pixel values
            
            # Convert back to PIL Image
            cloaked_img = (cloaked_img * 255).astype(np.uint8)
            cloaked_img = Image.fromarray(cloaked_img)
            
            # Add watermark
            watermarked_img = self.add_watermark(cloaked_img)
            
            # Save the final image
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(self.output_dir, f"{name}_cloaked{ext}")
            
            # Convert to RGB if saving as JPEG
            if ext.lower() in ('.jpg', '.jpeg'):
                watermarked_img = watermarked_img.convert('RGB')
            
            watermarked_img.save(output_path)
            return output_path
            
        except Exception as e:
            raise Exception(f"Cloaking failed: {str(e)}")

    def perform_cloaking(self):
        if not self.original_image_path:
            messagebox.showwarning("No Image Selected", "Please select an image first.")
            return

        try:
            self.update_status("Processing: Applying cloaking perturbations and watermark...")
            self.cloak_button.config(state=tk.DISABLED)
            self.root.update()
            
            # Call our cloaking function
            cloaked_path = self.cloak_image(self.original_image_path)
            
            if cloaked_path and os.path.exists(cloaked_path):
                self.cloaked_image_path = cloaked_path
                self.display_cloaked_image(cloaked_path)
                
                self.update_status(f"Cloaking complete! Saved to: {cloaked_path}")
                messagebox.showinfo(
                    "Success",
                    f"Image cloaked and watermarked successfully!\nSaved to:\n{cloaked_path}"
                )
            else:
                messagebox.showerror("Error", "Cloaking failed to save the image.")
                self.update_status("Cloaking failed.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during cloaking: {str(e)}")
            self.update_status(f"Error: {str(e)}")
        finally:
            self.cloak_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = CloakImageApp(root)
    root.mainloop()
