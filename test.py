import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time

def list_available_devices():
    """List all available RealSense devices and their capabilities"""
    ctx = rs.context()
    devices = ctx.query_devices()
    if devices.size() == 0:
        print("No RealSense devices found!")
        return []
    
    device_list = []
    for dev in devices:
        print(f"\nFound device: {dev.get_info(rs.camera_info.name)}")
        print(f"    Serial number: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"    Firmware version: {dev.get_info(rs.camera_info.firmware_version)}")
        
        # Check for IMU support
        sensors = dev.query_sensors()
        print("    Available sensors:")
        has_imu = False
        for sensor in sensors:
            print(f"        - {sensor.get_info(rs.camera_info.name)}")
            if sensor.is_motion_sensor():
                has_imu = True
        print(f"    IMU Support: {'Yes' if has_imu else 'No'}")
        
        device_list.append(dev)
    return device_list

def configure_streams(config, width=640, height=480):
    """Configure streams with error handling"""
    success = True
    try:
        # Try to enable color stream
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, 15)

        print("Color stream configured successfully")
    except Exception as e:
        print(f"Failed to configure color stream: {e}")
        success = False

    try:
        # Try to enable depth stream
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 15)
        
        print("Depth stream configured successfully")
    except Exception as e:
        print(f"Failed to configure depth stream: {e}")
        success = False

    # Try to enable IMU streams
    try:
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        print("IMU streams configured successfully")
    except Exception as e:
        print(f"Failed to configure IMU streams: {e}")
        print("Device might not support IMU")
        return success, False

    return success, True

def get_orientation(accel_data, gyro_data=None):
    """Calculate orientation from IMU data using complementary filter"""
    # Calculate roll and pitch from accelerometer data
    x, y, z = accel_data.x, accel_data.y, accel_data.z
    
    # Calculate angles from accelerometer
    roll = math.atan2(y, z) * 180.0 / math.pi
    pitch = math.atan2(-x, math.sqrt(y * y + z * z)) * 180.0 / math.pi
    
    return roll, pitch

def main():
    # First, list available devices
    print("\nSearching for RealSense devices...")
    devices = list_available_devices()
    if not devices:
        print("No devices found. Please check connection and permissions.")
        return

    # Initialize pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()

    print("\nConfiguring streams...")
    streams_success, imu_success = configure_streams(config)
    if not streams_success:
        print("Failed to configure basic streams")
        return

    # Try to start streaming with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\nAttempting to start pipeline (attempt {attempt + 1}/{max_retries})...")
            profile = pipeline.start(config)
            print("Pipeline started successfully!")
            break
        except RuntimeError as e:
            print(f"Failed to start pipeline: {e}")
            if attempt < max_retries - 1:
                print("Waiting before retry...")
                time.sleep(2)
            else:
                print("Max retries reached. Please check your camera connection and permissions.")
                return

    try:
        # Create align object to align depth frames to color frames
        align = rs.align(rs.stream.color)
        
        while True:
            # Wait for a coherent set of frames
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"Error getting frames: {e}")
                continue

            # Process IMU data if available
            if imu_success:
                # Get IMU data
                accel = None
                gyro = None
                for frame in frames:
                    if frame.is_motion_frame():
                        if frame.profile.stream_type() == rs.stream.accel:
                            accel = frame.as_motion_frame().get_motion_data()
                        if frame.profile.stream_type() == rs.stream.gyro:
                            gyro = frame.as_motion_frame().get_motion_data()

            # Align frames
            aligned_frames = align.process(frames)
            # aligned_frames= frames
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            print(depth_frame)
            color_frame = aligned_frames.get_color_frame()
            infrared_frame = aligned_frames.get_infrared_frame()
            
            if not depth_frame or not color_frame or not infrared_frame:
                print("Missing depth or color frame")
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            infrared_image = np.asanyarray(infrared_frame.get_data())
            
            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            infrared_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(infrared_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )

            # Stack all images horizontally
            images = np.hstack((color_image, depth_colormap, infrared_colormap))
            
            # Add IMU data overlay if available
            if imu_success and accel:
                roll, pitch = get_orientation(accel, gyro)
                
                imu_text = f"Roll: {roll:>6.1f} deg"
                cv2.putText(images, imu_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2, cv2.LINE_AA)
                imu_text = f"Pitch: {pitch:>6.1f} deg"
                cv2.putText(images, imu_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Add accelerometer raw data
                accel_text = f"Accel: X:{accel.x:>6.1f} Y:{accel.y:>6.1f} Z:{accel.z:>6.1f}"
                cv2.putText(images, accel_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if gyro:
                    gyro_text = f"Gyro: X:{gyro.x:>6.1f} Y:{gyro.y:>6.1f} Z:{gyro.z:>6.1f}"
                    cv2.putText(images, gyro_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\nStopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()