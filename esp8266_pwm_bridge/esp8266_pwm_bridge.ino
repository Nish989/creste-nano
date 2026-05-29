/*
 * ESP8266 PWM Bridge for RC Car
 *
 * Receives serial commands from Jetson via USB and outputs
 * standard 50Hz RC servo PWM on two pins.
 *
 * Wiring:
 *   D1 (GPIO5)  → ESC signal (throttle)
 *   D2 (GPIO4)  → Servo signal (steering)
 *   GND         → Common ground with servo/ESC
 *   5V/VIN      → BEC power (red wire from servo & ESC)
 *
 * Serial protocol (500000 baud):
 *   S<pulse_us>\n   — set steering pulse (e.g. "S1500\n")
 *   T<pulse_us>\n   — set throttle pulse (e.g. "T1600\n")
 *   N\n             — set both to neutral (1500us)
 *   P\n             — ping, responds "OK\n"
 *   L1\n / L0\n     — LED on / off (recording indicator)
 */

#include <Servo.h>

#define ESC_PIN     D1  // GPIO5
#define SERVO_PIN   D2  // GPIO4
#define LED_PIN     D4  // GPIO2 — onboard LED (active LOW)

#define NEUTRAL_US  1500
#define MIN_US      1000
#define MAX_US      2000
#define BAUD_RATE   500000

Servo escServo;
Servo steerServo;

String inputBuffer = "";

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // OFF at boot (active LOW)

  escServo.attach(ESC_PIN, MIN_US, MAX_US);
  steerServo.attach(SERVO_PIN, MIN_US, MAX_US);

  // Start at neutral
  escServo.writeMicroseconds(NEUTRAL_US);
  steerServo.writeMicroseconds(NEUTRAL_US);

  Serial.println("ESP8266_PWM_BRIDGE_READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
      // Prevent buffer overflow
      if (inputBuffer.length() > 20) {
        inputBuffer = "";
      }
    }
  }
}

void processCommand(String cmd) {
  char type = cmd.charAt(0);

  switch (type) {
    case 'T': // Throttle
    case 't': {
      int us = cmd.substring(1).toInt();
      if (us >= MIN_US && us <= MAX_US) {
        escServo.writeMicroseconds(us);
      }
      break;
    }
    case 'S': // Steering
    case 's': {
      int us = cmd.substring(1).toInt();
      if (us >= MIN_US && us <= MAX_US) {
        steerServo.writeMicroseconds(us);
      }
      break;
    }
    case 'N': // Neutral both
    case 'n':
      escServo.writeMicroseconds(NEUTRAL_US);
      steerServo.writeMicroseconds(NEUTRAL_US);
      break;
    case 'P': // Ping
    case 'p':
      Serial.println("OK");
      break;
    case 'L': // LED (recording indicator)
    case 'l': {
      int val = cmd.substring(1).toInt();
      digitalWrite(LED_PIN, val ? LOW : HIGH);  // active LOW
      break;
    }
  }
}
