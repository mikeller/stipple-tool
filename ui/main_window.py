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


class ProcessingThread(QThread):
    """Worker thread for processing models."""

    progress = pyqtSignal(int)
    batch_progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        selected_color,
        output_path,
        current_file,
        radius,
        depth,
        spheres_per_mm2,
    ):
        super().__init__()
        self.selected_color = selected_color
        self.radius = radius
        self.depth = depth
        self.spheres_per_mm2 = spheres_per_mm2
        self.output_path = output_path
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
            from core.manifold_stipple_processor import ManifoldStippleProcessor

            class _CancelledError(Exception):
                pass

            self.progress.emit(10)
            self.status.emit("Starting manifold stippling...")

            def on_status(msg: str):
                if self._cancelled:
                    raise _CancelledError()
                self.status.emit(msg)

            processor = ManifoldStippleProcessor()
            result = processor.process(
                step_file=self.current_file,
                output_path=self.output_path,
                target_color=self.selected_color,
                sphere_radius=self.radius,
                sphere_depth=self.depth,
                spheres_per_mm2=self.spheres_per_mm2,
                status_callback=on_status,
            )

            if result and Path(result).exists():
                self.progress.emit(100)
                self.finished.emit(True, f"Successfully saved to: {result}")
                self.status.emit("Done")
            else:
                self.finished.emit(False, "Stippling failed - no output generated")

        except _CancelledError:
            self.finished.emit(False, "__cancelled__")
        except Exception as e:
            error_msg = f"Error during processing: {e!s}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEP Stippling Tool")
        self.setGeometry(100, 100, 1000, 800)

        self.step_loader = STEPLoader()
        self.color_analyzer = ColorAnalyzer()

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
        
        # Stipple size (radius)
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setMinimum(0.1)
        self.size_spin.setMaximum(10.0)
        self.size_spin.setValue(1.4)
        self.size_spin.setSuffix(" mm")
        self.size_spin.setSingleStep(0.1)
        config_layout.addRow("Sphere Radius:", self.size_spin)
        
        # Stipple depth
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setMinimum(0.05)
        self.depth_spin.setMaximum(5.0)
        self.depth_spin.setValue(0.6)
        self.depth_spin.setSuffix(" mm")
        self.depth_spin.setSingleStep(0.05)
        config_layout.addRow("Cut Depth:", self.depth_spin)
        
        # Stipple density (spheres per mm²)
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setMinimum(0.01)
        self.density_spin.setMaximum(5.0)
        self.density_spin.setValue(0.5)
        self.density_spin.setDecimals(2)
        self.density_spin.setSuffix(" /mm²")
        self.density_spin.setSingleStep(0.05)
        config_layout.addRow("Density:", self.density_spin)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Preview group
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlainText(
            "Stipple Preview:\n\n"
            "Size: Radius of each spherical indentation\n"
            "Depth: How deep the indentations will be\n"
            "Density: How many stipples per unit area\n\n"
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
                <li>Export modified models as mesh (STL/3MF/OBJ)</li>
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
        """Load a STEP file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open STEP File",
            "",
            "STEP Files (*.step *.stp);;All Files (*)"
        )
        
        if file_path:
            ext = Path(file_path).suffix.lower()
            if ext not in [".step", ".stp"]:
                QMessageBox.warning(self, "Warning", "Only STEP files are supported. Please use a STEP file.")
                return

            if self.step_loader.load(file_path):
                self.current_file = file_path
                self.file_label.setText(f"Loaded: {Path(file_path).name}")
                self.analyze_colors()
            else:
                QMessageBox.critical(self, "Error", "Failed to load model")
    
    def analyze_colors(self):
        """Analyze colors in the loaded model."""
        if self.step_loader.get_model() is None:
            QMessageBox.warning(self, "Warning", "Please load a model file first")
            return
        
        try:
            self.color_list.clear()
            self.target_color_combo.clear()

            model_data = self.step_loader.get_model()
            color_groups = self.color_analyzer.extract_colors_from_model(model_data)
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
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Stippled Model As",
            "",
            "STL Files (*.stl);;3MF Files (*.3mf);;OBJ Files (*.obj)",
        )
        
        if file_path:
            out_path = Path(file_path)
            if out_path.suffix.lower() not in {".stl", ".3mf", ".obj"}:
                out_path = out_path.with_suffix(".stl")
            self.output_path = str(out_path)
            self.output_path_label.setText(f"Output: {out_path.name}")
    
    def process_model(self):
        """Process the model with stippling."""
        if self.step_loader.get_model() is None:
            QMessageBox.warning(self, "Warning", "Please load a model file first")
            return
        
        if not self.output_path:
            QMessageBox.warning(self, "Warning", "Please select an output file")
            return
        
        # Start processing thread
        selected_color = self.target_color_combo.currentData()
        if not selected_color:
            selected_color = self.target_color_combo.currentText()
        
        self.processing_thread = ProcessingThread(
            selected_color,
            self.output_path,
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
        if not success and message == "__cancelled__":
            self.progress_bar.setValue(0)
            self.status_label.setText("Cancelled")
            return

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
