import threading

from flask import Flask, render_template, request


import modbus_tk.defines as cst
from modbus_tk import modbus_tcp, hooks

import numpy as np
import time
import base64

from utility import *
from udp_img.udp_server import ImageReceiver


class WebApp:
    def __init__(self):
        self.app = Flask(__name__, static_folder='templates')
        self.input_str = None

        @self.app.route('/')
        def index():
            name = "Printing Machine UI"
            introduction = "This is a test of control UI for printing machine."
            contact_info = "Contact me at: Wenjie.Cui@br-automation.com"
            return render_template('index.html', name=name, introduction=introduction,
                                   contact_info=contact_info)

        @self.app.route('/submit', methods=['POST'])
        def submit():
            user_input = request.form['user_input']
            cmd_return_text = ""

            try:
                result = master.execute(slave=1, function_code=cst.WRITE_SINGLE_REGISTER, starting_address=8,
                                        output_value=2)

                cmd_return_text = "Write operation successful. Returned result: {}\n".format(result)
            except Exception as e:
                cmd_return_text = "Error: {}\n".format(e)

            input_text = "Your cmd " + user_input + " has been processed! \n"

            return input_text + cmd_return_text

        @self.app.route('/view_img', methods=['GET'])
        def view_customers():
            global img_base64
            return render_template('view_img.html', image_base64=img_base64)


    def run(self):
        self.app.run(debug=False)


def run_udp_receiver(ix: int):
    global img_base64
    receiver = ImageReceiver(server_port=9999)
    interval = 1  # 设置循环时间
    while True:
        start_time = time.time()

        img = receiver.receive_image()

        img_base64 = base64.b64encode(img).decode()
        task_time = time.time() - start_time
        wait_time = interval - task_time

        # if wait_time > 0:
        #     time.sleep(wait_time)

    #receiver.close()


if __name__ == '__main__':
    height, width, channels = 800, 2048, 3
    img_list = [np.zeros((height, width, channels), dtype=np.uint8),
                np.zeros((height, width, channels), dtype=np.uint8)]

    img_base64 = None
    t = threading.Thread(target=run_udp_receiver, args=(0,))
    t.start()
    web_app = WebApp()

    master = modbus_tcp.TcpMaster(host="127.0.0.1", port=502, timeout_in_sec=5.0)

    web_app.run()
