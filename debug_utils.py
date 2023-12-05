import threading
import traceback
import sys


class CustomThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super(CustomThread, self).run()
        except Exception as e:
            self.exc = e
            self.exc_traceback = sys.exc_info()[2]
            print("An exception was raised in the thread.")
            traceback.print_exception(type(self.exc), self.exc, self.exc_traceback)
