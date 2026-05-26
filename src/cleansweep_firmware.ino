// This is the only code on the Arduino
// Receives the velocity commands from the Pi, drives motors, reads ultrasonic sensors and MPU-6050, and sends back state.
//
// Serial protocol:
//    Pi -> Arduino : V<linear>,<angular>\n
//    Arduino -> Pi : S<dist_front>,<dist_rear>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>\n
//                    distances in cm; accel in m/s^2; gyro in rad/s

#include <Arduino.h>
#include <Wire.h>
#include <Servo.h>

// === Servo ===
Servo scanServo;
#define SERVO_PIN 10


// === Motor pins ===
// Each motor channel needs only 1 direction pin
#define MOTOR_STBY    3    // Standby 
#define MOTOR_R_PWM   5    // PWMA — right motors (M2, M3)
#define MOTOR_L_PWM   6    // PWMB — left motors (M1, M4)
#define MOTOR_R_DIR   7    // AIN1 — HIGH=forward, LOW=reverse
#define MOTOR_L_DIR   8    // BIN1 — HIGH=forward, LOW=reverse

// === Ultrasonic pins ===
// Front sensor
#define TRIG_FRONT  13
#define ECHO_FRONT  12
// Rear sensor
#define TRIG_REAR   A1
#define ECHO_REAR   A2

// MPU-6050 
#define MPU_ADDR      0x68   // Sensor's Address
#define REG_PWR_MGMT  0x6B   // Power setting register
#define REG_ACCEL     0x3B   // Where the accelerometer reading starts
#define REG_GYRO      0x43   // Where the rotation reading starts

// The sensor gives raw numbers, therefore they need to be multiplied to get real units 
//   Accel -> m/s^2
//   Gyro -> rad/s
#define ACCEL_SCALE  (9.81f   / 16384.0f)
#define GYRO_SCALE   (3.14159265f / (180.0f * 131.0f))

// === Timing and safety constants ===
#define SERIAL_BAUD         115200
#define SEND_INTERVAL_MS    50      // 20 Hz output rate
#define COMMAND_TIMEOUT_MS  500     // Stop motors if Pi stops talking
// wait for 20000UL (around 3.4m) for a sensor echo before giving up
#define ULTRASONIC_TIMEOUT  20000UL

// === Motor command state ===
float cmd_linear  = 0.0f;
float cmd_angular = 0.0f;
unsigned long last_cmd_time = 0;


// === MPU-6050 helpers ===

void mpuInit() {
  Wire.setClock(100000);         // slow speed so the sensor does not fail

  // Wake up the sensor
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_PWR_MGMT);
  Wire.write(0x00);
  Wire.endTransmission(true);
  delay(50);

  // asking the sensor for its ID
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);
  Wire.endTransmission(true);     
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  uint8_t who = Wire.available() ? Wire.read() : 0xFF;
  Serial.print("D MPU who_am_i=0x"); Serial.println(who, HEX);
}

// Reads 3 values (x,y,z) from the sensor and scales them to real units
// Each value is read separately 
void mpuRead3(uint8_t reg, float scale, float out[3]) {
  for (int i = 0; i < 3; i++) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg + (i * 2));
    Wire.endTransmission(true);         // FULL STOP instead of repeated start
    uint8_t got = Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)2);
    if (got == 2) {
      int16_t raw = ((int16_t)Wire.read() << 8) | Wire.read();
      out[i] = (float)raw * scale;
    } else {
      out[i] = 0.0f;
    }
  }
}


// === Ultrasonic helper ===

// Returns distance in cm, or 999.0 if no echo is returned within the timeout limit
float readUltrasonic(int trig, int echo) {
  // Send a short pulse to start a measurement
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  // Measuring how long an echo takes to come back
  long duration = pulseIn(echo, HIGH, ULTRASONIC_TIMEOUT);
  if (duration == 0) return 999.0f;     // No echo within the time limit
  return duration * 0.0343f / 2.0f;     // Converting the echo to distance
}


// === Motor helpers ===

void setMotors(float linear, float angular) {
  const float WHEEL_BASE = 0.125f;      // Distance between the wheels
  const float MAX_PWM    = 255.0f;      // The highest speed value
  const int   MIN_PWM    = 60;          // The lowest speed the car will move at
  const float LEFT_SCALE = 0.96f;       // Limits the left side so the car drives straight

  // Working out how fast each side should turn
  float left_speed  = linear - (angular * WHEEL_BASE / 2.0f);
  float right_speed = linear + (angular * WHEEL_BASE / 2.0f);

  // If either side is asking to go faster than full speed, scale both speeds down
  float max_val = max(abs(left_speed), abs(right_speed));
  if (max_val > 1.0f) {
    left_speed  /= max_val;
    right_speed /= max_val;
  }

  // Turning the speeds into motor power values
  int left_pwm  = (int)(abs(left_speed)  * MAX_PWM * LEFT_SCALE);
  int right_pwm = (int)(abs(right_speed) * MAX_PWM);

  // Making sure the motor power values is withing the robot's lowest speed
  if (left_pwm  > 0 && left_pwm  < MIN_PWM)  left_pwm  = MIN_PWM;
  if (right_pwm > 0 && right_pwm < MIN_PWM)  right_pwm = MIN_PWM;

  // Setting the direction and speed for each side 
  digitalWrite(MOTOR_L_DIR, left_speed >= 0.0f ? HIGH : LOW);
  analogWrite(MOTOR_L_PWM, left_pwm);

  digitalWrite(MOTOR_R_DIR, right_speed >= 0.0f ? HIGH : LOW);
  analogWrite(MOTOR_R_PWM, right_pwm);
}

void stopMotors() {
  analogWrite(MOTOR_L_PWM, 0);
  analogWrite(MOTOR_R_PWM, 0);
}



// === Setup ===

void setup() {
  Serial.begin(SERIAL_BAUD);

  Wire.begin();
  mpuInit();

  // Motor pins
  pinMode(MOTOR_STBY, OUTPUT);
  digitalWrite(MOTOR_STBY, HIGH);  // Enabling the motor driver
  pinMode(MOTOR_L_PWM, OUTPUT);
  pinMode(MOTOR_L_DIR, OUTPUT);
  pinMode(MOTOR_R_PWM, OUTPUT);
  pinMode(MOTOR_R_DIR, OUTPUT);

  // Ultrasonic pins
  pinMode(TRIG_FRONT, OUTPUT);
  pinMode(ECHO_FRONT, INPUT);
  pinMode(TRIG_REAR,  OUTPUT);
  pinMode(ECHO_REAR,  INPUT);

  stopMotors();
  
  // Making sure the servo is pointing directly straight
  scanServo.attach(SERVO_PIN);
  scanServo.write(90);

}



// === Main loop ===

unsigned long last_send = 0;
String incoming = "";

void loop() {
  // Reading any command from the Raspberry Pi
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {      // Full line has arrived
      if (incoming.startsWith("V")) {     // Speed command
        int comma = incoming.indexOf(',');
        if (comma > 1) {
          cmd_linear  = incoming.substring(1, comma).toFloat();
          cmd_angular = incoming.substring(comma + 1).toFloat();
          last_cmd_time = millis();
        }
      } else if (incoming.startsWith("A")) {      // Servo angle command
        int angle = incoming.substring(1).toInt();
        angle = constrain(angle, 30, 150);
        scanServo.write(angle);
      }
      // Clearing the line, so it is ready for the next line
      incoming = "";      
    } else {
      // The line is still being built
      incoming += c;      
    }
  }

  // === Safety: stop motors if Pi has gone silent ===
  if (millis() - last_cmd_time > COMMAND_TIMEOUT_MS) {
    cmd_linear  = 0.0f;
    cmd_angular = 0.0f;
  }

  // === Drive motors ===
  if (cmd_linear == 0.0f && cmd_angular == 0.0f) stopMotors();
  else setMotors(cmd_linear, cmd_angular);

  // === Send sensor state at 20 Hz ===
  if (millis() - last_send >= SEND_INTERVAL_MS) {
    last_send = millis();

    float dist_front = readUltrasonic(TRIG_FRONT, ECHO_FRONT);
    float dist_rear  = readUltrasonic(TRIG_REAR,  ECHO_REAR);

    float accel[3], gyro[3];
    mpuRead3(REG_ACCEL, ACCEL_SCALE, accel);  // x,y,z  acceleration (a)
    mpuRead3(REG_GYRO,  GYRO_SCALE,  gyro);   // x,y,z  rotation (g)

    // Sending one line: S<dist_front>,<dist_rear>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>
    Serial.print("S");
    Serial.print(dist_front, 2); Serial.print(",");
    Serial.print(dist_rear,  2); Serial.print(",");
    Serial.print(accel[0],   4); Serial.print(",");
    Serial.print(accel[1],   4); Serial.print(",");
    Serial.print(accel[2],   4); Serial.print(",");
    Serial.print(gyro[0],    4); Serial.print(",");
    Serial.print(gyro[1],    4); Serial.print(",");
    Serial.println(gyro[2],  4);
  }
}
