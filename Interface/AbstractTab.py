from PyQt6.QtWidgets import QApplication, QWidget


class AbstractTab(QWidget):
  def __init__(self, parent_class, tab_name):
    super().__init__()
    self.parent_class = parent_class
    self.tab_name = tab_name

  def tabSelected(self) -> None:
    raise NotImplementedError("Please Implement this method")
