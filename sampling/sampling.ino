// Original code by Wihan Booyse
// Modified by Ian de Villiers

#include <Wire.h>
#include "SparkFun_ADXL345.h"

#define ACC (0x53)
#define REG_POWER_CTL 0x2D
#define ACC_TO_READ (2)
#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))
#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit))

int y;

int regAddress = 0x34;

unsigned long lastTime;

void setup()
{
  sbi(ADCSRA, ADPS2);
  cbi(ADCSRA, ADPS1);
  cbi(ADCSRA, ADPS0);
  Serial.begin(1000000);
  Serial.println();
  delay(500);
  setupADXL();
  Serial.print("dt");
  Serial.print(",");
  Serial.println("acc");
}

void loop()
{
  unsigned long now = micros();
  unsigned long delta = now - lastTime;
  if (delta >= 556)               // ~1500x per second
  {
    lastTime = now;
    readADXLY();
    Serial.print(delta);
    Serial.print(",");
    Serial.print(y);
    Serial.println();
  }
}

///CUSTOM FUNCTIONS

//Preparing the ADXL unit
void setupADXL()
{
  Wire.begin();
  Wire.setClock(400000);
  writeTo(REG_POWER_CTL, 0x08);
  writeTo(ADXL345_BW_RATE, ADXL345_BW_1600);
  writeTo(ADXL345_DATA_FORMAT, B00000011);
}

// Reading the y-acceleration
bool readADXLY()
{
  byte buff[ACC_TO_READ];

  Wire.beginTransmission(ACC);
  Wire.write(regAddress);
  Wire.endTransmission(ACC);

  Wire.beginTransmission(ACC);
  Wire.requestFrom(ACC, ACC_TO_READ);

  int i = 0;
  while (Wire.available() && i < 2)
  {
    buff[i] = Wire.read();
    i++;
  }
  Wire.endTransmission();

  if (i == 2)  // valid read
  {
    y = (((int)buff[1]) << 8) | buff[0];
    return true;
  }
  return false;
}

// Write to helper function
void writeTo(byte _address, byte _val) {
  Wire.beginTransmission(ACC);
  Wire.write(_address);
  Wire.write(_val);
  Wire.endTransmission();
}
