#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

/* This driver reads raw data from the BNO055

   Connections
   ===========
   Connect SCL to analog 5
   Connect SDA to analog 4
   Connect VDD to 3.3V DC
   Connect GROUND to common ground

   History
   =======
   2015/MAR/03  - First release (KTOWN)
*/

#define BNO055_SAMPLERATE_DELAY_MS (100)
static constexpr double DEGREES_PER_RADIAN =
      (180.0 / 3.141592653589793238463); ///< Degrees per radian for conversion

Adafruit_BNO055 bno = Adafruit_BNO055(-1, 0x28, &Wire);

void setup(void) {
	Serial.begin(115200);

	while(!Serial); // Wait for serial port to open

	Serial.println("Orientation Sensor Raw Data Test"); Serial.println("");

	if (!bno.begin()) { // Initialize BNO055
		Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
		while(1);
	}
  bno.setMode(OPERATION_MODE_AMG); // Set the BNO055 to only AMG mode - no fusion/calibration
	delay(1000);

	bno.setExtCrystalUse(true);
}

void loop(void) {
	#ifdef MAGNETO_CAL
	// Possible vector values can be:
	// - VECTOR_ACCELEROMETER - m/s^2
	// - VECTOR_MAGNETOMETER  - uT
	// - VECTOR_GYROSCOPE     - rad/s
	// - VECTOR_EULER         - degrees
	// - VECTOR_LINEARACCEL   - m/s^2
	// - VECTOR_GRAVITY       - m/s^2
	#ifdef ACCELEROMETER
		imu::Vector<3> vector = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
	#elif defined(MAGNETOMETER)
		imu::Vector<3> vector = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
	#endif
	

	Serial.print(vector.x());
	Serial.print(",");
	Serial.print(vector.y());
	Serial.print(",");
	Serial.print(vector.z());
	Serial.println();

	#elif defined(ADAFRUIT_CAL)
	imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
	imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
	imu::Vector<3> mag = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

	// 'Raw' values to match expectation of MOtionCal
    Serial.print("Raw:");
    Serial.print(int(accel.x()*8192/9.8)); Serial.print(",");
    Serial.print(int(accel.y()*8192/9.8)); Serial.print(",");
    Serial.print(int(accel.z()*8192/9.8)); Serial.print(",");
    Serial.print(int(gyro.x()*DEGREES_PER_RADIAN*16)); Serial.print(",");
    Serial.print(int(gyro.y()*DEGREES_PER_RADIAN*16)); Serial.print(",");
    Serial.print(int(gyro.z()*DEGREES_PER_RADIAN*16)); Serial.print(",");
    Serial.print(int(mag.x()*10)); Serial.print(",");
    Serial.print(int(mag.y()*10)); Serial.print(",");
    Serial.print(int(mag.z()*10)); Serial.println("");

    // unified data
    Serial.print("Uni:");
    Serial.print(accel.x()); Serial.print(",");
    Serial.print(accel.y()); Serial.print(",");
    Serial.print(accel.z()); Serial.print(",");
    Serial.print(gyro.x(), 4); Serial.print(",");
    Serial.print(gyro.y(), 4); Serial.print(",");
    Serial.print(gyro.z(), 4); Serial.print(",");
    Serial.print(mag.x()); Serial.print(",");
    Serial.print(mag.y()); Serial.print(",");
    Serial.print(mag.z()); Serial.println("");

	#endif

	delay(BNO055_SAMPLERATE_DELAY_MS);
}
