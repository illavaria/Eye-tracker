from PyQt6.QtGui import QColor, QPalette
from Interface.AbstractTab import AbstractTab


class SettingsTab(AbstractTab):
    def __init__(self, parent_class, tab_name):
        super().__init__(parent_class, tab_name)
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 0))
        self.setPalette(palette)
    def tab_selected(self):
        return