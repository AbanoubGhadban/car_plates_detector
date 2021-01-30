from threading import Thread, Lock
import time
import RPi.GPIO as GPIO
import pyrebase

SERVO_PIN = 3
SERVO_ON_DUTY_CYCLE = 3
SERVO_OFF_DUTY_CYCLE = 10

GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm=GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

config = {
  "apiKey": "AIzaSyCBxiiFFd6aq6ufkFUxevosM2g9suvIjHE",
  "authDomain": "collegepark-1.firebaseapp.com",
  "databaseURL": "https://collegepark-1-default-rtdb.firebaseio.com",
  "storageBucket": "collegepark-1.appspot.com"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

class PlatesMonitor:
    def __init__(self, white_list_numbers):
        self.white_list_numbers = white_list_numbers
        self.last_seen = {}
        self.currently_available = []
        self.lock = Lock()
        self.stopped = False

    def start(self):
        Thread(target=self._run,args=()).start()

    def _run(self):
        servo_on_time = -1
        while True:
            cur_time = int(time.time()*1000)
            if servo_on_time != -1 and (cur_time - servo_on_time) > 1000:
                GPIO.output(SERVO_PIN, False)
                pwm.ChangeDutyCycle(0)
                servo_on_time = -1

            if self.stopped:
                pwm.stop()
                GPIO.cleanup()
            
            available = []
            for n, t in self.last_seen.items():
                if (cur_time - t) < 5000:
                    available.append(n)
            available.sort()

            if self.currently_available != available:
                plate_no = available[0] if len(available) > 0 else ""
                db.child("car detect").child("car plate").set(plate_no)

            if len(self.currently_available) == 0 or len(available) == 0:
                servo_on_time = cur_time
                GPIO.output(SERVO_PIN, True)
                if len(self.currently_available) == 0:
                    pwm.ChangeDutyCycle(SERVO_ON_DUTY_CYCLE)
                else:
                    pwm.ChangeDutyCycle(SERVO_OFF_DUTY_CYCLE)
            time.sleep(.01)

    def plate_detected(self, detected_number):
        if detected_number not in self.white_list_numbers:
            return
        try:
            self.lock.acquire()
            self.last_seen[detected_number] = int(time.time()*1000)
        finally:
            self.lock.release()

    def stop(self):
        self.stopped = True
