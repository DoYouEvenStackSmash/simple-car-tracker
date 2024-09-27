import numpy as np
import pygame

class KalmanFilter:
    def __init__(self, dt, process_noise_std, measurement_noise_std):
        # Time step
        self.dt = dt

        # State vector [x, y, v_x, v_y, a_x, a_y, theta]
        self.x = np.zeros((7, 1))

        # State covariance matrix
        self.P = np.eye(7) * 100  # Initial large uncertainty

        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 0, 1, 0, self.dt, 0, 0],
            [0, 0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Control matrix (for acceleration input)
        self.B = np.zeros((7, 3))
        self.B[4, 0] = dt
        self.B[5, 1] = dt
        self.B[6,2] = dt
        # Measurement matrix (for position, acceleration, and rotation)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # x position
            [0, 1, 0, 0, 0, 0, 0],  # y position
            [0, 0, 0, 0, 1, 0, 0],  # a_x
            [0, 0, 0, 0, 0, 1, 0],  # a_y
            [0, 0, 0, 0, 0, 0, 1],  # theta (rotation)
        ])

        # Process noise covariance matrix (Q)
        self.Q = np.eye(7) * process_noise_std**2

        # Measurement noise covariance matrix (R)
        self.R = np.eye(5) * measurement_noise_std**2

    def predict(self, u):
        """ Predict the next state and update the state covariance """
        # Predict state
        # print(self.B.shape)
        new_x = np.dot(self.F, self.x) + np.dot(self.B, u)

        # Predict state covariance
        new_P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return new_x,new_P

    def update(self, nx,nP, z):
        """ Update the state with measurement z """
        y = z - np.dot(self.H, nx)  # Measurement residual
        S = np.dot(self.H, np.dot(nP, self.H.T)) + self.R  # Residual covariance
        K = np.dot(np.dot(nP, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)  # Update state estimate
        self.P = nP - np.dot(K, np.dot(self.H, nP))  # Update covariance matrix

def draw_uncertainty_ellipse(screen, P, x_est, screen_size):
    """Draws the uncertainty ellipse and its axes based on the covariance matrix P and the estimated state x."""
    # Extract the covariance submatrix for position (2x2 block)
    cov = P[:2,:2]
    # print(P)

    # Eigenvalues and eigenvectors for ellipse axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Semi-major and semi-minor axes
    a = np.sqrt(eigenvalues[0])  # semi-major axis
    b = np.sqrt(eigenvalues[1])  # semi-minor axis

    # Dynamic scaling factor to fit the ellipse within the screen size
    max_screen_dim = min(screen_size[0], screen_size[1])  # Choose the smaller screen dimension
    scale_factor = 0.2* max_screen_dim / (2 * np.sqrt(eigenvalues[0]))  # Scales according to largest eigenvalue

    # Scale the axes for visualization
    width = 2 * a * scale_factor  # Major axis length
    height = 2 * b * scale_factor  # Minor axis length

    # Ellipse angle from the eigenvector direction
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 1])#* 180 / np.pi
    print(angle)
    # Center of the ellipse is the estimated position
    center = ((x_est[0, 0]), (x_est[1, 1]))

    # Compute the distance from the center to the foci
    if a > b:
        c = np.sqrt(a**2 - b**2) * scale_factor
    else:
        c = 0  # In the case of a circle, the foci are the same as the center
    print(eigenvectors,'eigen')
    # eigenvectors*=angle
    # Direction of the semi-major and semi-minor axes (from eigenvectors)
    major_axis_direction = eigenvectors[:, 0]  # Corresponds to the largest eigenvalue
    minor_axis_direction = eigenvectors[:, 1]  # Corresponds to the smallest eigenvalue
    major_axis_direction[0]=np.cos(angle+np.pi)
    major_axis_direction[1] = np.sin(angle)
    minor_axis_direction[0] = np.cos(angle-np.pi/2)
    minor_axis_direction[1] = np.sin(angle+np.pi/2)
    # Compute the two foci of the ellipse
    foci1 = (center[0] + (c * major_axis_direction[0]), 
             center[1] + (c * major_axis_direction[1]))
    foci2 = (center[0] - (c * major_axis_direction[0]), 
             center[1] - (c * major_axis_direction[1]))

    # Draw the ellipse
    # ellipse_rect = pygame.Rect(center[0] - width / 2, center[1] - height / 2, width, height)
    # pygame.draw.ellipse(screen, (0, 0, 255), ellipse_rect, 2)

    # Draw the foci as small circles
    pygame.draw.circle(screen, (255, 0, 0), foci1, 5)  # First focus
    pygame.draw.circle(screen, (255, 0, 0), foci2, 5)  # Second focus

    # print(major_axis_direction,'maj')
    # exit()
    # major_axis_direction[0]=np.cos(major_axis_direction[0]*angle)
    # major_axis_direction[1] = np.sin(major_axis_direction[1]*angle)
    # minor_axis_direction[0] = np.cos(minor_axis_direction[0]*angle)
    # minor_axis_direction[1] = np.sin(minor_axis_direction[1]*angle)
    # Compute the endpoints of the major and minor axes for drawing
    major_axis_start = (center[0] + (a * scale_factor * major_axis_direction[0]), 
                        center[1] + (a * scale_factor * major_axis_direction[1]))
                        
    major_axis_end = (center[0] - (a * scale_factor * major_axis_direction[0]), 
                      center[1] - (a * scale_factor * major_axis_direction[1]))

    minor_axis_start = (center[0] + (b * scale_factor * minor_axis_direction[0]), 
                        center[1] + (b * scale_factor * minor_axis_direction[1]))
    minor_axis_end = (center[0] - (b * scale_factor * minor_axis_direction[0]), 
                      center[1] - (b * scale_factor * minor_axis_direction[1]))
    
    # pygame.draw.polygon(screen,(0,0,255),[major_axis_start,minor_axis_start,major_axis_end,minor_axis_end],width=2)
    # pygame.draw.ellipse(screen, (0, 0, 255), ellipse_rect, 2)
    # Draw the major and minor axes
    pygame.draw.line(screen, (0, 255, 0), major_axis_start, major_axis_end, 2)  # Major axis in green
    pygame.draw.line(screen, (255, 255, 0), minor_axis_start, minor_axis_end, 2)  # Minor axis in yellow



import math
MAX_SPEED=60
class SimpleCarModel:
    def __init__(self, L, min_phi=-math.pi/2, max_phi=math.pi/2, x=200, y=300, theta=0, s=0.0):
        # Initialize the car parameters
        self.L = L  # Distance between the front and rear axles
        self.min_phi = min_phi  # Minimum steering angle
        self.max_phi = max_phi  # Maximum steering angle
        self.x = x  # Initial x position
        self.y = y  # Initial y position
        self.theta = theta  # Initial orientation (angle in radians)
        self.s = s  # Initial speed of the car
        self.clock=0

    def update_state(self, u_s, u_phi, dt):
        # Ensure u_s is within the bounds [-1, 1]
        u_s = max(min(u_s, MAX_SPEED), -MAX_SPEED)
        
        # Ensure u_phi is within the bounds [min_phi, max_phi]
        u_phi = max(min(u_phi, self.max_phi), self.min_phi)
        
        # Kinematic equations for the car's motion
        x_dot = u_s * math.cos(self.theta)
        y_dot = u_s * math.sin(self.theta)
        theta_dot = u_s / self.L * math.tan(u_phi)
        
        # Update the state over the time step dt
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.theta += theta_dot * dt
        self.clock+=dt

    def get_state(self):
        # Return the current state as (x, y, theta)
        return self.x, self.y,self.s * math.cos(self.theta), self.s * math.sin(self.theta), self.theta
    def accelerate_to_speed(self, target_speed, acceleration_rate, dt):
        """
        Accelerates the car to the target speed at a given rate.
        target_speed: Desired target speed (bounded by [-1, 1])
        acceleration_rate: Rate of speed increase per time step
        dt: Time step
        """
        # Ensure the target speed is within bounds [-1, 1]
        target_speed = max(min(target_speed, MAX_SPEED), -MAX_SPEED)
        
        # Increase speed linearly towards the target speed
        if self.s < target_speed:
            self.s = min(self.s + acceleration_rate * dt, target_speed)
        elif self.s > target_speed:
            self.s = max(self.s - acceleration_rate * dt, target_speed)

    def decelerate_to_stop(self, deceleration_rate, dt):
        """
        Decelerates the car to a stop at a given rate.
        deceleration_rate: Rate of speed decrease per time step
        dt: Time step
        """
        # Decrease speed towards zero
        if self.s > 0:
            self.s = max(self.s - deceleration_rate * dt, 0)
        elif self.s < 0:
            self.s = min(self.s + deceleration_rate * dt, 0)
import time
# np.linalg.norm
def draw_paths(screen):
    """Draws the actual and estimated paths on the screen."""
    screen.fill((0, 0, 0))  # Clear the screen
    # print(actual_path)
    # print(estimated_path)
    # Draw the actual path in green
    for i in range(2, len(actual_path)):
        pygame.draw.line(screen, (0, 255, 0), actual_path[i - 1], actual_path[i], 4)

    # Draw the estimated path in red
    for i in range(1, len(corrupted_path)):
        pygame.draw.circle(screen, (255, 255, 255), corrupted_path[i - 1],2)#, corrupted_path[i], 1)
        # Draw the estimated path in red
    for i in range(1, len(estimated_path)-1):
        pygame.draw.line(screen, (255, 0, 0), estimated_path[i - 1], estimated_path[i], 4)
    # pygame.display.flip()
actual_path,estimated_path,corrupted_path=[],[],[]

import matplotlib.pyplot as plt

def main():
     # Define car parameters
    L = 5.1  # Wheelbase of the car

    # Create a SimpleCarModel instance
    car = SimpleCarModel(L)

    # Example action input (speed and steering angle)
    u_phi = 0.01  # Steering angle in radians (bounded between [-pi/2, pi/2])
    dt = 0.1  # Time step of 100ms

    # Example of accelerating to a target speed
    target_speed = 50  # Target speed (bounded between [-1, 1])
    acceleration_rate = 10  # Speed increase per time step

    # for _ in range(20):  # Simulate for 20 steps
    #     car.accelerate_to_speed(target_speed, acceleration_rate, dt)
    #     car.update_state(car.s, u_phi, dt)
    #     state = car.get_state()
    #     z = (state[0],state[1], acceleration_rate,0,0)

    #     print(f"Accelerating -> x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}, speed: {state[3]:.2f}")

    # # Example of decelerating to stop
    # deceleration_rate = 0.1  # Speed decrease per time step

    # for _ in range(10):  # Simulate for 10 steps
    #     car.decelerate_to_stop(deceleration_rate, dt)
    #     car.update_state(car.s, u_phi, dt)
    #     state = car.get_state()
    #     print(f"Decelerating -> x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}, speed: {state[3]:.2f}")

    pygame.init()
    screen_size = (800, 600)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Kalman Filter with Uncertainty Ellipses")
    clock = pygame.time.Clock()

    # Initialize the Kalman filter
    dt = 0.05  # Time step (100 ms)
    kf = KalmanFilter(dt, process_noise_std=0.1, measurement_noise_std=4)

    # Control input (acceleration)
    u = np.array([[0.00], [0.00],[0]])
    path = []
    kpf = []
    # Measurement (position, acceleration, rotation)
    # u = np.array([[300], [400],[0], [0], [0]])
    # state = car.get_state()
    # kf
    # u = np.array((state[0],state[1], acceleration_rate,0,0))
    # kf.predict(u)
    # kf.update(state)
    t=0
    sim_time=20
    running = True
    location = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    location -= 1
                if event.key == pygame.K_RIGHT:
                    location += 1


        if t>sim_time:
            break
    # for _ in range(20):  # Simulate for 20 steps
        # car.accelerate_to_speed(target_speed, acceleration_rate, dt)
        # car.update_state(car.s, u_phi, dt)
        # state = car.get_state()
        # print(f"Accelerating -> x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}, speed: {state[3]:.2f}")
        screen.fill((255, 255, 255))  # Clear screen with white background

        # Kalman filter predict step
        nx,nP=kf.predict(u)
        # print(u)
        car.accelerate_to_speed(target_speed, acceleration_rate, dt)
        car.update_state(car.s, u_phi*location, dt)
        state = car.get_state()
        print(state)
        actual_path.append((state[0],state[1]))
        # print(kf.x)

        # estimated_path.append()
        # Kalman filter update step with the measurement
        # state[0:2]+=
        z = np.array(state)
        # z[0:2]+=
        npt = np.random.rand()*np.sqrt(12)
        ng = np.random.randn() * np.pi
        z[0:2] += npt * np.array((np.cos(ng),np.sin(ng)))
        kf.update(nx,nP,z)
        print(nx)
        # nx,nP=kf.predict(z)
        x_fut = kf.F @ kf.x

        estimated_path.append((x_fut[0,0],x_fut[1,1]))
        corrupted_path.append((z[0],z[1]))
        # print(kf.x)
        
        print(state)
        # z = kf.x[:,0]
        u = np.array([[z[2]],[z[3]],[z[4]]])

        # kpf.append(kf.x)
        draw_paths(screen)

        # Draw the uncertainty ellipse
        
        draw_uncertainty_ellipse(screen, kf.P, x_fut, screen_size)

        pygame.display.flip()  # Update the display
        t+=dt
        time.sleep(dt)
        # clock.tick(60)  # 60 FPS
    # error = estimated_path
    pygame.quit()

if __name__ == "__main__":
    main()
