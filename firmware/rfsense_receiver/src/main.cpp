#include <Arduino.h>
#include <ArduinoJson.h>
#include "../include/iot_display.h"
#include "../include/iot_pins.h"

constexpr unsigned long serialSpeed = 115200;
constexpr size_t maxLineBytes = 511;
constexpr unsigned long alertHoldMs = 3000;
constexpr unsigned long buzzMs = 150;

enum class Alert { Idle, Info, Warning, Critical };

char line[maxLineBytes];
size_t lineLength = 0;
bool skipLine = false;
bool displayReady = false;
Alert activeAlert = Alert::Idle;
unsigned long lastAlert = 0;

void showAlert(Alert alert, const char* severity, const char* label, const String& details) {
  digitalWrite(IoTPins::LED_3, alert == Alert::Idle);
  digitalWrite(IoTPins::LED_2, alert == Alert::Info);
  digitalWrite(IoTPins::LED_1, alert >= Alert::Warning);

  if (alert == Alert::Critical && activeAlert != Alert::Critical) {
    tone(IoTPins::BUZZER, 2500, buzzMs);
  } else if (alert != Alert::Critical && activeAlert == Alert::Critical) {
    noTone(IoTPins::BUZZER);
  }

  activeAlert = alert;
  lastAlert = millis();
  if (displayReady) {
    String text(label);
    iotDisplayLines(severity, text.substring(0, 21), text.substring(21, 42), details);
  }
}

void handleEvent() {
  JsonDocument event;
  if (deserializeJson(event, line, lineLength, DeserializationOption::NestingLimit(4))) {
    Serial.println("ERR invalid JSON");
    return;
  }
  if (event["type"] != "detection") {
    Serial.println("ERR expected a detection event");
    return;
  }

  const char* severity = event["alert"] | "";
  Alert alert;
  if (event["alert"] == "green") {
    alert = Alert::Idle;
  } else if (event["alert"] == "info" || event["alert"] == "yellow") {
    alert = Alert::Info;
  } else if (event["alert"] == "warning" || event["alert"] == "red") {
    alert = Alert::Warning;
  } else if (event["alert"] == "critical") {
    alert = Alert::Critical;
  } else {
    Serial.println("ERR unknown alert");
    return;
  }

  String details;
  if (event["center_mhz"].is<float>() && event["confidence"].is<float>()) {
    details = String(event["center_mhz"].as<float>(), 1) + " MHz " +
              String(event["confidence"].as<float>() * 100, 0) + "%";
  }
  showAlert(alert, severity, event["label"] | "", details);
  Serial.println("OK");
}

void setup() {
  pinMode(IoTPins::LED_1, OUTPUT);
  pinMode(IoTPins::LED_2, OUTPUT);
  pinMode(IoTPins::LED_3, OUTPUT);
  pinMode(IoTPins::BUZZER, OUTPUT);
  digitalWrite(IoTPins::BUZZER, LOW);
  Serial.begin(serialSpeed);
  displayReady = iotInitDisplay();
  showAlert(Alert::Idle, "RF monitor", "Waiting for alert", "");
}

void loop() {
  if (activeAlert != Alert::Idle && millis() - lastAlert >= alertHoldMs) {
    showAlert(Alert::Idle, "RF monitor", "Waiting for alert", "");
  }

  for (int remaining = 256; remaining > 0 && Serial.available(); --remaining) {
    int incomingByte = Serial.read();
    if (incomingByte == '\n') {
      if (skipLine) {
        Serial.println("ERR invalid serial line");
      } else if (lineLength) {
        handleEvent();
      }
      lineLength = 0;
      skipLine = false;
    } else if (!skipLine) {
      if (lineLength == sizeof(line) || incomingByte == '\0') {
        skipLine = true;
      } else {
        line[lineLength++] = incomingByte;
      }
    }
  }
}
