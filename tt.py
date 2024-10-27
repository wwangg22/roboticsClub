import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

def create_point_cloud_from_depth(depth_image, color_image, intrinsic, depth_scale=1000.0):
    # Create Open3D image objects
    depth = o3d.geometry.Image(depth_image)
    color = o3d.geometry.Image(color_image)
    
    # Create RGBD image pair
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=depth_scale,
        depth_trunc=5.0,
        convert_rgb_to_intensity=False)
    
    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.ppx,
            cy=intrinsic.ppy
        )
    )
    
    return pcd

def main():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get camera intrinsics
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = depth_profile.get_intrinsics()
    
    # Create Open3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud")
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Create point cloud
            pcd = create_point_cloud_from_depth(depth_image, color_image, intrinsics)
            
            # Flip the point cloud for better visualization
            #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            
            # Update visualization
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            # Break loop if window is closed
            if not vis.poll_events():
                break
            
    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()