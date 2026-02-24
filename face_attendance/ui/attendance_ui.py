#!/usr/bin/env python3
"""
UI untuk sistem absensi wajah
"""

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import cv2
import threading
import time
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.camera import CameraManager
from face_recognition.detector import FaceDetector
from face_recognition.recognizer import FaceRecognizer
from models.database import DatabaseManager

class AttendanceUI:
    def __init__(self):
        self.camera = None
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.db = DatabaseManager()
        self.is_running = False
        self.current_frame = None
        self._camera_label_photo = None
        self._after_time_id = None
        self._after_video_id = None
        
        # Setup window
        self.root = ctk.CTk()
        self.root.title("🎭 Absensi Wajah")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        self.setup_ui()
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass
        
    def setup_ui(self):
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="🎭 Absensi Wajah", 
                                  font=("Arial", 24, "bold"))
        title_label.pack(pady=20)
        
        # Content frame
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left panel - Camera
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Camera title
        camera_title = ctk.CTkLabel(left_panel, text="📷 Kamera", 
                                   font=("Arial", 16, "bold"))
        camera_title.pack(pady=10)
        
        # Camera display
        self.camera_label = ctk.CTkLabel(left_panel, text="Kamera belum aktif", 
                                        width=400, height=300,
                                        fg_color="gray", corner_radius=10)
        self.camera_label.pack(pady=10)
        
        # Camera controls
        camera_controls = ctk.CTkFrame(left_panel)
        camera_controls.pack(pady=10)
        
        self.start_camera_btn = ctk.CTkButton(camera_controls, text="▶️ Mulai Kamera",
                                            command=self.start_camera, width=120)
        self.start_camera_btn.grid(row=0, column=0, padx=5)
        
        self.stop_camera_btn = ctk.CTkButton(camera_controls, text="⏹️ Hentikan Kamera",
                                           command=self.stop_camera, width=120,
                                           state="disabled")
        self.stop_camera_btn.grid(row=0, column=1, padx=5)
        
        # Right panel - Attendance info
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Status title
        status_title = ctk.CTkLabel(right_panel, text="📊 Status Absensi", 
                                   font=("Arial", 16, "bold"))
        status_title.pack(pady=10)
        
        # Current time
        self.time_label = ctk.CTkLabel(right_panel, text="Waktu: --:--:--", 
                                      font=("Arial", 14))
        self.time_label.pack(pady=5)
        
        # Status
        self.status_label = ctk.CTkLabel(right_panel, text="Status: Siap", 
                                        font=("Arial", 14))
        self.status_label.pack(pady=5)
        
        # Recognized person
        self.person_label = ctk.CTkLabel(right_panel, text="Orang: -", 
                                        font=("Arial", 14))
        self.person_label.pack(pady=5)
        
        # Confidence
        self.confidence_label = ctk.CTkLabel(right_panel, text="Confidence: -", 
                                           font=("Arial", 14))
        self.confidence_label.pack(pady=5)
        
        # Attendance button
        self.attendance_btn = ctk.CTkButton(right_panel, text="✅ Lakukan Absensi",
                                          command=self.do_attendance, width=200, height=40,
                                          state="disabled")
        self.attendance_btn.pack(pady=20)
        
        # Bottom panel - Recent attendances
        bottom_panel = ctk.CTkFrame(main_frame)
        bottom_panel.pack(fill="x", padx=20, pady=10)
        
        recent_title = ctk.CTkLabel(bottom_panel, text="📝 Absensi Terbaru", 
                                   font=("Arial", 14, "bold"))
        recent_title.pack(pady=10)
        
        # Recent attendances list
        self.recent_list = ctk.CTkTextbox(bottom_panel, height=100, width=600)
        self.recent_list.pack(pady=10)
        
        # Control buttons
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(pady=10)
        
        back_btn = ctk.CTkButton(control_frame, text="🏠 Kembali ke Menu",
                                 command=self.back_to_menu, width=150)
        back_btn.grid(row=0, column=0, padx=5)
        
        refresh_btn = ctk.CTkButton(control_frame, text="🔄 Refresh",
                                   command=self.refresh_recent, width=100)
        refresh_btn.grid(row=0, column=1, padx=5)
        
        # Update time
        self.update_time()
        
        # Load recent attendances
        self.refresh_recent()
        
    def update_time(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.configure(text=f"Waktu: {current_time}")
        try:
            self._after_time_id = self.root.after(1000, self.update_time)
        except Exception:
            pass
        
    def start_camera(self):
        try:
            self.camera = CameraManager()
            self.camera.start()
            self.is_running = True
            
            # Update UI
            self.start_camera_btn.configure(state="disabled")
            self.stop_camera_btn.configure(state="normal")
            self.attendance_btn.configure(state="normal")
            self.status_label.configure(text="Status: Kamera aktif")
            
            # Start video loop
            self.update_video()
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memulai kamera: {e}")
            self.status_label.configure(text="Status: Error kamera")
            
    def stop_camera(self):
        self.is_running = False
        
        if self.camera:
            self.camera.stop()
            self.camera = None
            
        # Update UI
        self.start_camera_btn.configure(state="normal")
        self.stop_camera_btn.configure(state="disabled")
        self.attendance_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Kamera dihentikan")
        self.camera_label.configure(image="", text="Kamera dihentikan")
        
    def update_video(self):
        if not self.is_running or not self.camera:
            return
            
        try:
            frame = self.camera.get_frame()
            if frame is not None:
                self.current_frame = frame.copy()
                
                # Detect faces
                faces = self.detector.detect_faces(frame)
                
                # Draw rectangles and show info
                for face in faces:
                    x1, y1, x2, y2 = face['bbox']
                    confidence = face['confidence']
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Show confidence
                    cv2.putText(frame, f"{confidence:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convert to PhotoImage for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image, ImageTk
                image = Image.fromarray(frame_rgb)
                
                # Resize to fit display
                display_width = 400
                display_height = 300
                image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=image)
                
                self.camera_label.configure(image=photo, text="")
                self._camera_label_photo = photo
                
                # Update status
                if len(faces) > 0:
                    self.status_label.configure(text=f"Status: {len(faces)} wajah terdeteksi")
                else:
                    self.status_label.configure(text="Status: Tidak ada wajah")
                    
        except Exception as e:
            print(f"Error update video: {e}")
            
        # Schedule next update
        try:
            self._after_video_id = self.root.after(30, self.update_video)
        except Exception:
            pass
        
    def do_attendance(self):
        if self.current_frame is None:
            messagebox.showwarning("Peringatan", "Tidak ada frame kamera!")
            return
            
        try:
            self.status_label.configure(text="Status: Memproses...")
            
            # Detect faces
            faces = self.detector.detect_faces(self.current_frame)
            
            if len(faces) == 0:
                messagebox.showwarning("Peringatan", "Tidak ada wajah terdeteksi!")
                self.status_label.configure(text="Status: Tidak ada wajah")
                return
                
            if len(faces) > 1:
                messagebox.showwarning("Peringatan", "Terlalu banyak wajah! Hanya 1 wajah yang diperbolehkan.")
                self.status_label.configure(text="Status: Terlalu banyak wajah")
                return
                
            # Get face region
            face = faces[0]
            x1, y1, x2, y2 = face['bbox']
            face_image = self.current_frame[y1:y2, x1:x2]
            
            if face_image.size == 0:
                messagebox.showerror("Error", "Gagal mendapatkan area wajah!")
                self.status_label.configure(text="Status: Error")
                return
                
            # Recognize face
            result = self.recognizer.recognize_face(face_image)
            
            if result['recognized']:
                employee_id = result['employee_id']
                confidence = result['confidence']
                
                # Get employee info
                employee = self.db.get_employee_by_id(employee_id)
                
                if employee:
                    name = employee['name']
                    
                    # Record attendance
                    self.db.record_attendance(employee_id, confidence)
                    
                    # Update UI
                    self.person_label.configure(text=f"Orang: {name}")
                    self.confidence_label.configure(text=f"Confidence: {confidence:.2f}")
                    self.status_label.configure(text="Status: Absensi berhasil!")
                    
                    # Show success message
                    messagebox.showinfo("Sukses", f"Absensi berhasil!\nNama: {name}\nConfidence: {confidence:.2f}")
                    
                    # Refresh recent attendances
                    self.refresh_recent()
                    
                else:
                    messagebox.showerror("Error", "Karyawan tidak ditemukan di database!")
                    self.status_label.configure(text="Status: Karyawan tidak ditemukan")
                    
            else:
                self.person_label.configure(text="Orang: Tidak dikenal")
                self.confidence_label.configure(text="Confidence: -")
                self.status_label.configure(text="Status: Wajah tidak dikenal")
                messagebox.showwarning("Peringatan", "Wajah tidak dikenali!\nSilakan registrasi terlebih dahulu.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Gagal melakukan absensi: {e}")
            self.status_label.configure(text="Status: Error absensi")
            
    def refresh_recent(self):
        try:
            # Clear current content
            self.recent_list.delete("1.0", tk.END)
            
            # Get recent attendances (last 10)
            recent_attendances = self.db.get_recent_attendances(limit=10)
            
            if recent_attendances:
                for attendance in recent_attendances:
                    employee = self.db.get_employee_by_id(attendance['employee_id'])
                    if employee:
                        name = employee['name']
                        time = attendance['attendance_time']
                        confidence = attendance['confidence']
                        
                        self.recent_list.insert(tk.END, f"📅 {time} - {name} (Confidence: {confidence:.2f})\n")
            else:
                self.recent_list.insert(tk.END, "Belum ada absensi hari ini.")
                
        except Exception as e:
            self.recent_list.insert(tk.END, f"Error: {e}")
            
    def back_to_menu(self):
        self.stop_camera()
        try:
            if self._after_time_id is not None:
                self.root.after_cancel(self._after_time_id)
                self._after_time_id = None
            if self._after_video_id is not None:
                self.root.after_cancel(self._after_video_id)
                self._after_video_id = None
        except Exception:
            pass
        self.root.destroy()
        
        # Kembali ke main menu
        try:
            from main import main
            main()
        except Exception as e:
            print(f"Error kembali ke menu: {e}")
    
    def _on_close(self):
        self.back_to_menu()
            
    def run(self):
        self.root.mainloop()
        
if __name__ == "__main__":
    app = AttendanceUI()
    app.run()
