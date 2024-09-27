#!/usr/bin/env python

import rospy
import random
from sensor_msgs.msg import Imu
from std_msgs.msg import Header

def mock_imu_data():
    rospy.init_node('imu_node', anonymous=True)
    
    imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=10)
    
    use_sim = rospy.get_param('~use_sim', False)
    
    rate = rospy.Rate(10)  # 10 Hz
    
    while not rospy.is_shutdown():
        imu_msg = Imu()
        
        # Fill the IMU message with mock data
        imu_msg.header = Header()
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "imu_link"
        
        # Mock orientation data (quaternion)
        imu_msg.orientation.x = random.uniform(-1.0, 1.0)
        imu_msg.orientation.y = random.uniform(-1.0, 1.0)
        imu_msg.orientation.z = random.uniform(-1.0, 1.0)
        imu_msg.orientation.w = random.uniform(-1.0, 1.0)
        
        # Mock angular velocity
        imu_msg.angular_velocity.x = random.uniform(-0.5, 0.5)
        imu_msg.angular_velocity.y = random.uniform(-0.5, 0.5)
        imu_msg.angular_velocity.z = random.uniform(-0.5, 0.5)
        
        # Mock linear acceleration
        imu_msg.linear_acceleration.x = random.uniform(-9.8, 9.8)
        imu_msg.linear_acceleration.y = random.uniform(-9.8, 9.8)
        imu_msg.linear_acceleration.z = random.uniform(-9.8, 9.8)
        
        # If use_sim is true, modify data for simulation purposes
        if use_sim:
            imu_msg.angular_velocity.z += 0.1  # Small bias in simulation mode
        
        # Publish the mock IMU data
        imu_pub.publish(imu_msg)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        mock_imu_data()
    except rospy.ROSInterruptException:
        pass
