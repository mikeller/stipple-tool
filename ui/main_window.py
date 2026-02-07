"""
Main GUI window for the STEP Stippling Tool.
"""
import sys
from pathlib import Path
from typing import Optional
import traceback
import time

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QComboBox,
    QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QListWidget, QListWidgetItem,
    QTabWidget, QTextEdit
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor

from core.step_loader import STEPLoader
from core.color_analyzer import ColorAnalyzer
from core.stencil_processor import StencilStippleProcessor
from core.mesh_loader import MeshLoader
from core.mesh_color_analyzer import MeshColorAnalyzer


class ProcessingThread(QThread):
    """Worker thread for processing models."""

    progress = pyqtSignal(int)
    batch_progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        model_loader,
        color_analyzer,
        selected_color,
        output_path,
        pattern,
        model_type,
        current_file,
        size,
        depth,
        density,
    ):
        super().__init__()
        self.model_loader = model_loader
        self.color_analyzer = color_analyzer
        self.selected_color = selected_color
        self.size = size
        self.depth = depth
        self.density = density
        self.output_path = output_path
        self.pattern = pattern
        self.model_type = model_type
        self.current_file = current_file
        self._cancelled = False

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def cancel(self):
        """Request cancellation of the processing."""
        self._cancelled = True

    def run(self):
        """Execute the stippling process."""
        try:
            self.progress.emit(10)

            model_data = self.model_loader.get_model()
            if model_data is None:
                self.finished.emit(False, "Model not loaded")
                return

            self.progress.emit(20)

            face_indices = self.color_analyzer.get_faces_by_color(self.selected_color)
            if not face_indices:
                self.finished.emit(False, f"No faces found with color: {self.selected_color}")
                return

            self.progress.emit(40)
            self.status.emit("Stippling in progress...")

            if self.model_type == "mesh":
                self.finished.emit(False, "Mesh input is no longer supported. Please use a STEP file.")
                return

            if self.model_type == "step":
                self._run_step_stippling(model_data, face_indices)
                return

            self.finished.emit(False, f"Unknown model type: {self.model_type}")

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)

    def _run_step_stippling(self, model_data, face_indices):
        """Apply stippling to STEP model using stencil-based approach."""
        try:
            self.progress.emit(20)
            self.status.emit("Starting stencil stippling...")
            
            # Create processor
            processor = StencilStippleProcessor()
            
            # Use parameters passed from main window
            sphere_radius = self.size  # In mm
            sphere_depth = self.depth  # In mm
            density = self.density  # 0.0-1.0
            
            # Convert density to spheres_per_mm2
            # Density 0.8 should map to roughly 0.15 spheres/mm2 for good coverage
            # Scale: 0.0 → 0.0, 0.5 → 0.1, 0.8 → 0.15, 1.0 → 0.2
            spheres_per_mm2 = density * 0.2
            
            # Stencil approach parameters
            strip_count = 8  # Number of strips to divide geometry
            overlap = 0.3  # 30% overlap between strips
            batch_size = 2  # Small batches to avoid geometry fragmentation
            
            def on_status(msg: str):
                if self.is_cancelled():
                    raise RuntimeError("Processing cancelled by user")
                self.status.emit(msg)
                # Simulate progress updates based on status messages
                if "strip" in msg.lower():
                    # Extract strip number if possible and update progress
                    import re
                    match = re.search(r'(\d+)/(\d+)', msg)
                    if match:
                        curr, total = int(match.group(1)), int(match.group(2))
                        self.progress.emit(int(20 + (curr / total) * 70))

            def on_cancel_check():
                if self.is_cancelled():
                    raise RuntimeError("Processing cancelled by user")

            result_path = processor.process_step_with_stencil_stippling(
                step_file=self.current_file,
                output_path=self.output_path,
                target_color=self.selected_color,
                sphere_radius=sphere_radius,
                sphere_depth=sphere_depth,
                spheres_per_mm2=spheres_per_mm2,
                strip_count=strip_count,
                overlap=overlap,
                batch_size=batch_size,
                size_variation=True,  # Add variety in hole sizes (0.5x to 1.5x radius)
                status_callback=on_status,
                cancel_callback=on_cancel_check,
            )
            
            self.progress.emit(95)
            
            if result_path and Path(result_path).exists():
                self.progress.emit(100)
                self.finished.emit(True, f"Successfully saved to: {result_path}")
                self.status.emit("Done")
            else:
                self.finished.emit(False, "Incremental stippling failed - no output generated")

        except Exception as e:
            error_msg = f"Error in STEP stippling: {str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)

    def _run_mesh_stippling(self, model_data, face_indices):
        """Run stippling on mesh models and export as OBJ/STL."""
        self.finished.emit(False, "Mesh input is no longer supported. Please use a STEP file.")
        return


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEP Stippling Tool")
        self.setGeometry(100, 100, 1000, 800)

        self.step_loader = STEPLoader()
        self.color_analyzer = ColorAnalyzer()
        self.mesh_loader = MeshLoader()
        self.mesh_color_analyzer = MeshColorAnalyzer()

        self.active_loader = self.step_loader
        self.active_color_analyzer = self.color_analyzer
        self.model_type = "step"

        self.current_file = None
        self.output_path = None
        self.processing_thread = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        title = QLabel("STEP Stippling Tool - Add Texture to Tool Handles")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        main_layout.addWidget(title)

        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file loaded")
        file_layout.addWidget(self.file_label)

        load_btn = QPushButton("Load STEP File")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)

        main_layout.addLayout(file_layout)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        color_tab = self.create_color_tab()
        tabs.addTab(color_tab, "Color Analysis")

        stipple_tab = self.create_stipple_tab()
        tabs.addTab(stipple_tab, "Stipple Configuration")

        process_tab = self.create_process_tab()
        tabs.addTab(process_tab, "Process & Export")

        info_tab = self.create_info_tab()
        tabs.addTab(info_tab, "Information")

    def create_color_tab(self) -> QWidget:
        """Create the color analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        color_group = QGroupBox("Surface Colors")
        color_layout = QVBoxLayout()

        color_layout.addWidget(QLabel("Detected colors in the model:"))

        self.color_list = QListWidget()
        self.color_list.currentItemChanged.connect(self.on_color_selected)
        color_layout.addWidget(self.color_list)

        analyze_btn = QPushButton("Analyze Colors")
        analyze_btn.clicked.connect(self.analyze_colors)
        color_layout.addWidget(analyze_btn)

        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        target_group = QGroupBox("Target Color for Stippling")
        target_layout = QFormLayout()

        target_layout.addRow(QLabel("Select color:"), QLabel("(Choose from list above)"))

        self.target_color_combo = QComboBox()
        target_layout.addRow("Color:", self.target_color_combo)

        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_stipple_tab(self) -> QWidget:
        """Create the stipple configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        config_group = QGroupBox("Stipple Parameters")
        config_layout = QFormLayout()
        
        # Stipple size
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setMinimum(0.1)
        self.size_spin.setMaximum(10.0)
        self.size_spin.setValue(1.0)
        self.size_spin.setSuffix(" mm")
        self.size_spin.setSingleStep(0.1)
        config_layout.addRow("Stipple Size:", self.size_spin)
        
        # Stipple depth
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setMinimum(0.05)
        self.depth_spin.setMaximum(5.0)
        self.depth_spin.setValue(0.5)
        self.depth_spin.setSuffix(" mm")
        self.depth_spin.setSingleStep(0.05)
        config_layout.addRow("Stipple Depth:", self.depth_spin)
        
        # Stipple density
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setMinimum(0.01)
        self.density_spin.setMaximum(1.0)
        self.density_spin.setValue(0.8)
        self.density_spin.setSingleStep(0.05)
        config_layout.addRow("Stipple Density:", self.density_spin)
        
        # Pattern type
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["random", "grid", "hexagon"])
        self.pattern_combo.setCurrentText("hexagon")
        config_layout.addRow("Pattern Type:", self.pattern_combo)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Preview group
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlainText(
            "Stipple Preview:\n\n"
            "Size: Diameter of each stipple indentation\n"
            "Depth: How deep the indentations will be\n"
            "Density: How many stipples per unit area\n"
            "Pattern: Distribution pattern of stipples\n\n"
            "Current settings will be applied to selected surfaces."
        )
        preview_layout.addWidget(self.preview_text)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_process_tab(self) -> QWidget:
        """Create the processing and export tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Output path
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        output_path_layout = QHBoxLayout()
        self.output_path_label = QLabel("No output file selected")
        output_path_layout.addWidget(self.output_path_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output)
        output_path_layout.addWidget(browse_btn)
        
        output_layout.addLayout(output_path_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Processing status
        status_group = QGroupBox("Processing")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(10)
        status_layout.setContentsMargins(10, 10, 10, 10)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        status_layout.addWidget(self.progress_bar)

        self.batch_progress_label = QLabel("Batch progress")
        self.batch_progress_label.setMinimumHeight(20)
        self.batch_progress_label.show()
        status_layout.addWidget(self.batch_progress_label)

        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setMinimum(0)
        self.batch_progress_bar.setMaximum(100)
        self.batch_progress_bar.setTextVisible(True)
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setMinimumHeight(30)
        self.batch_progress_bar.show()
        status_layout.addWidget(self.batch_progress_bar)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        process_btn = QPushButton("Apply Stippling & Export")
        process_btn.clicked.connect(self.process_model)
        button_layout.addWidget(process_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel_processing)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_info_tab(self) -> QWidget:
        """Create the information tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        info_group = QGroupBox("About This Tool")
        info_layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
            <h2>STEP Stippling Tool</h2>
            <p>
                This tool adds gripping textures (stippling) to 3D models 
                in STEP format for 3D printing.
            </p>
            <h3>Features:</h3>
            <ul>
                <li>Load and analyze STEP files</li>
                <li>Detect surface colors</li>
                <li>Apply stippling textures to selected surfaces</li>
                <li>Customize stipple parameters</li>
                <li>Export modified models to STEP format</li>
            </ul>
            <h3>Workflow:</h3>
            <ol>
                <li>Load a STEP file</li>
                <li>Analyze surface colors</li>
                <li>Select which color to stipple</li>
                <li>Configure stipple parameters</li>
                <li>Process and export</li>
            </ol>
            <h3>Technical Details:</h3>
            <p>
                Stippling works by creating small hemispherical indentations
                on the surface, improving grip and texture.
            </p>
            <p><small>© 2026 STEP Stippling Tool</small></p>
        """)
        info_layout.addWidget(info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        widget.setLayout(layout)
        return widget
    
    def load_file(self):
        """Load a STEP file (only format with reliable color detection)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Model File",
            "",
            "STEP Files (*.step *.stp);;All Files (*)"
        )
        
        if file_path:
            ext = Path(file_path).suffix.lower()
            if ext in [".step", ".stp"]:
                loader = self.step_loader
                analyzer = self.color_analyzer
                self.model_type = "step"
            else:
                # Mesh formats (currently not supported for color detection)
                QMessageBox.warning(self, "Warning", "Only STEP files support multi-color detection. Please use a STEP file.")
                return

            if loader.load(file_path):
                self.current_file = file_path
                self.file_label.setText(f"Loaded: {Path(file_path).name}")
                self.active_loader = loader
                self.active_color_analyzer = analyzer
                self.analyze_colors()
            else:
                QMessageBox.critical(self, "Error", "Failed to load model")
    
    def analyze_colors(self):
        """Analyze colors in the loaded model."""
        if self.active_loader.get_model() is None:
            QMessageBox.warning(self, "Warning", "Please load a model file first")
            return
        
        try:
            self.color_list.clear()
            self.target_color_combo.clear()

            model_data = self.active_loader.get_model()
            color_groups = self.active_color_analyzer.extract_colors_from_model(model_data)
            colors = list(color_groups.keys())

            if not colors:
                colors = ["default"]  # Fallback to single color
                color_groups = {"default": list(range(len(model_data.get("faces", []))))}

            for color in colors:
                face_count = len(color_groups.get(color, []))
                label = f"{color} ({face_count} faces)"

                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, color)

                if color.startswith("#") and len(color) == 7:
                    swatch = QColor(color)
                    item.setBackground(swatch)
                    item.setForeground(self._color_text_color(swatch))

                self.color_list.addItem(item)
                self.target_color_combo.addItem(label, color)

            if self.target_color_combo.count() > 0:
                self.target_color_combo.setCurrentIndex(0)

            # Provide feedback about color detection
            color_count = len(colors)
            status_msg = f"Found {color_count} color(s)"
            
            if color_count == 1 and self.model_type == "mesh":
                status_msg += " (only 1 color - check if a STEP file exists with more colors)"
            
            self.status_label.setText(status_msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing colors: {str(e)}")
    
    def on_color_selected(self, current: QListWidgetItem, previous: Optional[QListWidgetItem]):
        """Sync color selection from list to combo."""
        if current is None:
            return
        color = current.data(Qt.ItemDataRole.UserRole)
        if color:
            index = self.target_color_combo.findData(color)
            if index >= 0:
                self.target_color_combo.setCurrentIndex(index)

    @staticmethod
    def _color_text_color(color: QColor) -> QColor:
        r = color.redF()
        g = color.greenF()
        b = color.blueF()
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return QColor("white") if luminance < 0.5 else QColor("black")

    def browse_output(self):
        """Browse for output file path."""
        filter_str = "STEP Files (*.step)"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Stippled Model As",
            "",
            filter_str,
        )
        
        if file_path:
            out_path = Path(file_path)
            if out_path.suffix.lower() != ".step":
                out_path = out_path.with_suffix(".step")
            self.output_path = str(out_path)
            self.output_path_label.setText(f"Output: {out_path.name}")
    
    def process_model(self):
        """Process the model with stippling."""
        if self.active_loader.get_model() is None:
            QMessageBox.warning(self, "Warning", "Please load a model file first")
            return
        
        if not self.output_path:
            QMessageBox.warning(self, "Warning", "Please select an output file")
            return
        
        # STEP-only output
        if self.output_path:
            out_path = Path(self.output_path)
            if out_path.suffix.lower() != ".step":
                self.output_path = str(out_path.with_suffix(".step"))
                self.output_path_label.setText(f"Output: {Path(self.output_path).name}")
        
        # Start processing thread
        selected_color = self.target_color_combo.currentData()
        if not selected_color:
            selected_color = self.target_color_combo.currentText()
        pattern = self.pattern_combo.currentText()
        
        self.processing_thread = ProcessingThread(
            self.active_loader,
            self.active_color_analyzer,
            selected_color,
            self.output_path,
            pattern,
            self.model_type,
            self.current_file,
            self.size_spin.value(),
            self.depth_spin.value(),
            self.density_spin.value(),
        )
        
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.batch_progress.connect(self.batch_progress_bar.setValue)
        self.processing_thread.status.connect(self.status_label.setText)
        self.processing_thread.finished.connect(self.on_processing_finished)
        
        self.status_label.setText("Processing...")
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)
        self.processing_thread.start()
    
    def cancel_processing(self):
        """Cancel the current processing."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
            self.status_label.setText("Cancelling...")
            self.processing_thread.wait(5000)
            if self.processing_thread.isRunning():
                # Thread did not stop gracefully - warn user but don't terminate
                # (terminate() is unsafe and can corrupt state)
                self.status_label.setText("Warning: Processing still running in background")
            else:
                self.status_label.setText("Cancelled")
            self.progress_bar.setValue(0)
    
    def on_processing_finished(self, success: bool, message: str):
        """Handle processing completion."""
        self.progress_bar.setValue(100 if success else 0)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.status_label.setText("Completed successfully!")
        else:
            QMessageBox.critical(self, "Error", message)
            self.status_label.setText("Processing failed")


def main():
    """Entry point for the GUI application."""
    app = __import__('PyQt6.QtWidgets', fromlist=['QApplication']).QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
