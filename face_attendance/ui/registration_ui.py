import customtkinter as ctk
import cv2
import threading
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
from face_recognition.trainer import FaceTrainer
from utils.camera import CameraManager
from models.database import DatabaseManager
from config.constants import FACE_POSES, SAMPLES_PER_POSE

class RegistrationUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("Face Attendance - Registrasi Karyawan")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        # Initialize components
        self.camera = CameraManager()
        self.trainer = FaceTrainer()
        self.db = DatabaseManager()
        
        self.is_capturing = False
        self.current_frame = None
        self._update_after_id = None
        
        self.setup_ui()
        self.start_camera()
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        except Exception:
            pass
        
    def setup_ui(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Camera and controls
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Camera view
        self.camera_label = ctk.CTkLabel(left_frame, text="", width=640, height=480)
        self.camera_label.pack(pady=10)
        
        # Progress bar
        self.progress_var = ctk.DoubleVar(value=0)
        self.progress_bar = ctk.CTkProgressBar(left_frame, variable=self.progress_var, width=600)
        self.progress_bar.pack(pady=5)
        
        # Status label
        self.status_label = ctk.CTkLabel(left_frame, text="Siap untuk registrasi", font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        # Control buttons
        button_frame = ctk.CTkFrame(left_frame)
        button_frame.pack(pady=10)
        
        self.start_button = ctk.CTkButton(button_frame, text="Mulai Registrasi", 
                                         command=self.start_registration, width=150)
        self.start_button.pack(side="left", padx=5)
        
        self.capture_button = ctk.CTkButton(button_frame, text="Ambil Foto", 
                                           command=self.capture_frame, width=150, state="disabled")
        self.capture_button.pack(side="left", padx=5)
        
        self.reset_button = ctk.CTkButton(button_frame, text="Reset", 
                                         command=self.reset_registration, width=150)
        self.reset_button.pack(side="left", padx=5)
        
        # Right panel - Employee info
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Title
        title_label = ctk.CTkLabel(right_frame, text="Registrasi Karyawan", 
                                  font=("Arial", 20, "bold"))
        title_label.pack(pady=20)
        
        # Employee form
        form_frame = ctk.CTkFrame(right_frame)
        form_frame.pack(fill="x", padx=20, pady=10)
        
        # Employee ID
        ctk.CTkLabel(form_frame, text="ID Karyawan:").pack(anchor="w", padx=10, pady=(10, 2))
        self.employee_id_entry = ctk.CTkEntry(form_frame, placeholder_text="Masukkan ID karyawan")
        self.employee_id_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # Employee name
        ctk.CTkLabel(form_frame, text="Nama Karyawan:").pack(anchor="w", padx=10, pady=(10, 2))
        self.employee_name_entry = ctk.CTkEntry(form_frame, placeholder_text="Masukkan nama lengkap")
        self.employee_name_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # Department
        ctk.CTkLabel(form_frame, text="Departemen:").pack(anchor="w", padx=10, pady=(10, 2))
        self.department_entry = ctk.CTkEntry(form_frame, placeholder_text="Masukkan departemen")
        self.department_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # Position
        ctk.CTkLabel(form_frame, text="Jabatan:").pack(anchor="w", padx=10, pady=(10, 2))
        self.position_entry = ctk.CTkEntry(form_frame, placeholder_text="Masukkan jabatan")
        self.position_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # Email
        ctk.CTkLabel(form_frame, text="Email:").pack(anchor="w", padx=10, pady=(10, 2))
        self.email_entry = ctk.CTkEntry(form_frame, placeholder_text="Masukkan email")
        self.email_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # Instructions
        instruction_frame = ctk.CTkFrame(right_frame)
        instruction_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(instruction_frame, text="Instruksi Pendaftaran:", 
                     font=("Arial", 14, "bold")).pack(pady=10)
        
        self.instruction_text = ctk.CTkLabel(instruction_frame, text="\n".join(FACE_POSES), 
                                           justify="left", wraplength=250)
        self.instruction_text.pack(pady=10)
        
        # Current pose indicator
        self.pose_label = ctk.CTkLabel(instruction_frame, text="Pose saat ini: -", 
                                      font=("Arial", 12, "bold"))
        self.pose_label.pack(pady=5)
        
        # Sample counter
        self.sample_label = ctk.CTkLabel(instruction_frame, text="Sample: 0/0", 
                                        font=("Arial", 12))
        self.sample_label.pack(pady=5)
        
    def start_camera(self):
        if self.camera.start():
            self.update_camera_feed()
        else:
            self.status_label.configure(text="Error: Kamera tidak dapat diakses")
    
    def update_camera_feed(self):
        if self.camera.is_running():
            frame = self.camera.get_frame()
            if frame is not None:
                self.current_frame = frame.copy()
                
                # Selalu gambar overlay deteksi (ROI + bbox + confidence)
                frame = self.trainer.draw_registration_ui(frame)
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                
                # Convert to PIL Image
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update label
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
            
            # Schedule next update
            try:
                self._update_after_id = self.root.after(30, self.update_camera_feed)
            except Exception:
                pass
    
    def start_registration(self):
        employee_id = self.employee_id_entry.get().strip()
        employee_name = self.employee_name_entry.get().strip()
        
        if not employee_id or not employee_name:
            self.status_label.configure(text="Error: ID dan Nama karyawan wajib diisi!")
            return
        
        # Check if employee already exists
        existing_employee = self.db.get_employee_by_id(employee_id)
        if existing_employee:
            self.status_label.configure(text="Error: ID karyawan sudah terdaftar!")
            return
        
        # Start registration process
        success = self.trainer.start_registration(employee_id, employee_name)
        if success:
            self.is_capturing = True
            self.start_button.configure(state="disabled")
            self.capture_button.configure(state="normal")
            self.status_label.configure(text="Registrasi dimulai. Ikuti instruksi pose!")
            self.update_progress()
    
    def capture_frame(self):
        if self.current_frame is None:
            return
        
        success, message = self.trainer.capture_frame(self.current_frame)
        
        if success:
            self.status_label.configure(text=f"✅ {message}")
            self.update_progress()
            
            if self.trainer.is_registration_complete():
                self.complete_registration()
        else:
            self.status_label.configure(text=f"❌ {message}")
    
    def update_progress(self):
        progress = self.trainer.get_progress()
        self.progress_var.set(progress / 100)
        
        current_pose = self.trainer.get_current_instruction()
        self.pose_label.configure(text=f"Pose saat ini: {current_pose}")
        
        samples_collected = self.trainer.samples_collected
        total_samples = len(FACE_POSES) * SAMPLES_PER_POSE
        self.sample_label.configure(text=f"Sample: {samples_collected}/{total_samples}")
    
    def complete_registration(self):
        department = self.department_entry.get().strip()
        position = self.position_entry.get().strip()
        email = self.email_entry.get().strip()
        
        # Update employee info
        self.db.update_employee(
            self.trainer.employee_id,
            department=department if department else None,
            position=position if position else None,
            email=email if email else None
        )
        
        success, message = self.trainer.complete_registration()
        
        if success:
            self.status_label.configure(text="✅ Registrasi berhasil!")
            self.is_capturing = False
            self.capture_button.configure(state="disabled")
            self.start_button.configure(state="normal")
            
            # Show success dialog
            self.show_success_dialog()
        else:
            self.status_label.configure(text=f"❌ Error: {message}")
    
    def reset_registration(self):
        self.trainer.reset_registration()
        self.is_capturing = False
        self.start_button.configure(state="normal")
        self.capture_button.configure(state="disabled")
        self.status_label.configure(text="Registrasi direset")
        self.progress_var.set(0)
        self.pose_label.configure(text="Pose saat ini: -")
        self.sample_label.configure(text="Sample: 0/0")
    
    def show_success_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Registrasi Berhasil")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Success message
        ctk.CTkLabel(dialog, text="🎉 Registrasi Berhasil!", 
                    font=("Arial", 18, "bold")).pack(pady=20)
        
        ctk.CTkLabel(dialog, text=f"Karyawan: {self.trainer.employee_name}", 
                    font=("Arial", 12)).pack(pady=5)
        
        ctk.CTkLabel(dialog, text=f"ID: {self.trainer.employee_id}", 
                    font=("Arial", 12)).pack(pady=5)
        
        ctk.CTkLabel(dialog, text=f"Total sample: {self.trainer.samples_collected}", 
                    font=("Arial", 12)).pack(pady=5)
        
        # OK button
        ctk.CTkButton(dialog, text="OK", command=dialog.destroy, width=100).pack(pady=20)
        
        # Clear form after success
        self.employee_id_entry.delete(0, 'end')
        self.employee_name_entry.delete(0, 'end')
        self.department_entry.delete(0, 'end')
        self.position_entry.delete(0, 'end')
        self.email_entry.delete(0, 'end')
    
    def run(self):
        self.root.mainloop()
    
    def on_closing(self):
        try:
            if self._update_after_id is not None:
                try:
                    self.root.after_cancel(self._update_after_id)
                except Exception:
                    pass
                self._update_after_id = None
        finally:
            self.camera.stop()
        self.root.destroy()

if __name__ == "__main__":
    app = RegistrationUI()
    app.root.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.run()
