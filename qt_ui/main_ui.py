import time
from threading import Thread

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QTableWidget
# PySide6-uic demo.ui -o ui_demo.py  # 生成UI
# from ui_demo import Ui_Demo
from printingUI import Ui_MainWindow

import modbus_tk.defines as cst
from modbus_tk import modbus_tcp, hooks

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()  # UI类的实例化()
        self.ui.setupUi(self)
        self.band()


    def band(self):
        # 常用信号与槽的绑定
        # self.ui.___ACTION___.triggered.connect(___FUNCTION___)
        # self.ui.___BUTTON___.clicked.connect(___FUNCTION___)
        # self.ui.___COMBO_BOX___.currentIndexChanged.connect(___FUNCTION___)
        # self.ui.___SPIN_BOX___.valueChanged.connect(___FUNCTION___)
        # 自定义信号.属性名.connect(___FUNCTION___)

        self.ui.stopPrintingButton.clicked.connect(self.handle_stopPrintingButton_click)  # function不要带括号！！！
        self.ui.modbusLabel.setText('no cmd')


    def handle_stopPrintingButton_click(self):
        try:
            result = master.execute(slave=1, function_code=cst.WRITE_SINGLE_REGISTER, starting_address=8,
                                    output_value=2)

            cmd_return_text = "Write operation successful. Returned result: {}\n".format(result)
            self.ui.modbusLabel.setText(cmd_return_text)
        except Exception as e:
            cmd_return_text = "Error: {}\n".format(e)
            self.ui.modbusLabel.setText(cmd_return_text)







if __name__ == '__main__':
    master = modbus_tcp.TcpMaster(host="127.0.0.1", port=502, timeout_in_sec=5.0)


    app = QApplication([])  # 启动一个应用
    window = MainWindow()  # 实例化主窗口
    window.show()  # 展示主窗口
    app.exec()  # 避免程序执行到这一行后直接退出
