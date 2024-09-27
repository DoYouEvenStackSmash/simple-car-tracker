# kalman_simulation.py
import pygame
import numpy as np

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((1500, 1000))
pygame.display.set_caption("Kalman Filter: Simulated Movement with Angular Velocity and Acceleration")

# Kalman Filter parameters
dt = 0.1  # Time step
A = np.array([[1, 0, dt, 0, 0, 0, 0],  # x position
              [0, 1, 0, dt, 0, 0, 0],  # y position
              [0, 0, 1, 0, 0, 0, 0],  # x velocity
              [0, 0, 0, 1, 0, 0, 0],  # y velocity
              [0, 0, 0, 0, 1, dt, 0.5 * dt**2],  # angular position (theta)
              [0, 0, 0, 0, 0, 1, dt],  # angular velocity (omega)
              [0, 0, 0, 0, 0, 0, 1]])  # angular acceleration (alpha)

# A = np.array([[1, 0, dt, 0, 0, 0, 0],  # x position
#               [0, 1, 0, dt, 0, 0, 0],  # y position
#               [0, 0, 1, 0, 0, 0, 0],  # x velocity
#               [0, 0, 0, 1, 0, 0, 0],  # y velocity
#               [0, 0, 0, 0, 1, dt, 0.5 * dt**2],  # angular position (theta)
#               [0, 0, 0, 0, 0, 1, dt],  # angular velocity (omega)
#               [0, 0, 0, 0, 0, 0, 1]])  # angular acceleration (alpha)

H = np.array([[1, 0, 0, 0, 0, 0, 0],  # We only observe x position
              [0, 1, 0, 0, 0, 0, 0]])  # We only observe y position

Q = np.eye(7) * 0.01  # Process noise covariance for x, y, and angular states
R = np.eye(2) * 5     # Measurement noise covariance for position
P = np.eye(7)         # Estimate error covariance

# Initial state: position (400, 300), velocity (0, 0), angular position (theta), angular velocity (omega), angular acceleration (alpha)
x = np.array([[400], [300], [0], [0], [0], [0], [0]])  # Including angular position, velocity, and acceleration

# Variables to track paths
actual_path = []
estimated_path = []

# Define maximum allowed angular acceleration
MAX_ALPHA = 2.0  # Example limit for angular acceleration

def limit_angular_acceleration(x):
    """Clamp the angular acceleration to the defined maximum limit."""
    x[6, 0] = np.clip(x[6, 0], -MAX_ALPHA, MAX_ALPHA)
    return x

def kalman_predict(x, P):
    # Prediction step
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q
    
    # Limit angular acceleration after prediction
    x_pred = limit_angular_acceleration(x_pred)
    
    return x_pred, P_pred

def kalman_update(x_pred, P_pred, z):
    # Update step
    S = H @ P_pred @ H.T + R  # Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
    y = z - (H @ x_pred)  # Innovation (residual)
    x_new = x_pred + K @ y  # Updated state estimate
    P_new = (np.eye(7) - K @ H) @ P_pred  # Updated estimate covariance
    
    # Limit angular acceleration after update
    # x_new = limit_angular_acceleration(x_new)
    
    return x_new, P_new

def simulated_mouse_position(t, amplitude=150, decay=0.1, frequency=1):
    # Simulated mouse movement as a decaying exponential multiplied by cosine
    x = amplitude * np.exp(-decay * t/10) * np.cos(frequency * t)
    if 3<t<4:
        y = amplitude * np.exp(-decay * t) * np.sign(np.sin(frequency * t*np.pi)) + np.random.randn()*np.sqrt(4)
    else:
        y = amplitude * np.exp(-decay * t) * np.sin(frequency * t*np.pi) + np.random.randn()*np.sqrt(4)
    
    return np.array([[800 - t/10*800], [300 + y]])  # Centered around (400, 300)
    px = pygame.mouse.get_pos()
    return np.array([[px[0]],[px[1]]])



def draw_uncertainty_ellipse(screen, P, x_est, screen_size=(800,600)):
    """Draws the uncertainty ellipse and its axes based on the covariance matrix P and the estimated state x."""
    # Extract the covariance submatrix for position (2x2 block)
    # print(P)
    cov = P[:2, :2]
    print(cov)
    # Eigenvalues and eigenvectors for ellipse axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    print(eigenvectors)
    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    print(eigenvectors)
    # Semi-major and semi-minor axes
    a = np.sqrt(eigenvalues[0])  # semi-major axis
    b = np.sqrt(eigenvalues[1])  # semi-minor axis
    # print(a)
    # Dynamic scaling factor to fit the ellipse within the screen size
    max_screen_dim = min(screen_size[0], screen_size[1])  # Choose the smaller screen dimension
    scale_factor = 0.2 * max_screen_dim / (2 * np.sqrt(eigenvalues[0]))  # Scales according to largest eigenvalue

    # Scale the axes for visualization
    width = 2 * a * scale_factor  # Major axis length
    height = 2 * b * scale_factor  # Minor axis length

    # Ellipse angle from the eigenvector direction
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
    # print(angle)
    # Center of the ellipse is the estimated position
    center = (int(x_est[0, 0]), int(x_est[1, 0]))

    # Compute the distance from the center to the foci
    if a > b:
        c = np.sqrt(a**2 - b**2) * scale_factor
    else:
        c = 0  # In the case of a circle, the foci are the same as the center

    # Direction of the semi-major and semi-minor axes (from eigenvectors)
    major_axis_direction = eigenvectors[:, 0]  # Corresponds to the largest eigenvalue
    minor_axis_direction = eigenvectors[:, 1]  # Corresponds to the smallest eigenvalue

    # Compute the two foci of the ellipse
    foci1 = (center[0] + int(c * major_axis_direction[0]), 
             center[1] + int(c * major_axis_direction[1]))
    foci2 = (center[0] - int(c * major_axis_direction[0]), 
             center[1] - int(c * major_axis_direction[1]))

    # Draw the ellipse
    ellipse_rect = pygame.Rect(center[0] - width / 2, center[1] - height / 2, width, height)
    pygame.draw.ellipse(screen, (0, 0, 255), ellipse_rect, 2)

    # Draw the foci as small circles
    pygame.draw.circle(screen, (255, 0, 0), foci1, 5)  # First focus
    pygame.draw.circle(screen, (255, 0, 0), foci2, 5)  # Second focus

    # Compute the endpoints of the major and minor axes for drawing
    major_axis_start = (center[0] + int(a * scale_factor * major_axis_direction[0]), 
                        center[1] + int(a * scale_factor * major_axis_direction[1]))
    major_axis_end = (center[0] - int(a * scale_factor * major_axis_direction[0]), 
                      center[1] - int(a * scale_factor * major_axis_direction[1]))

    minor_axis_start = (center[0] + int(b * scale_factor * minor_axis_direction[0]), 
                        center[1] + int(b * scale_factor * minor_axis_direction[1]))
    minor_axis_end = (center[0] - int(b * scale_factor * minor_axis_direction[0]), 
                      center[1] - int(b * scale_factor * minor_axis_direction[1]))

    # Draw the major and minor axes
    pygame.draw.line(screen, (0, 255, 0), major_axis_start, major_axis_end, 2)  # Major axis in green
    pygame.draw.line(screen, (255, 255, 0), minor_axis_start, minor_axis_end, 2)  # Minor axis in yellow
import math

class SimpleCarModel:
    def __init__(self, L, min_phi=-math.pi/2, max_phi=math.pi/2, x=300, y=400, theta=10.0, s=0.0):
        # Initialize the car parameters
        self.L = L  # Distance between the front and rear axles
        self.min_phi = min_phi  # Minimum steering angle
        self.max_phi = max_phi  # Maximum steering angle
        self.x = x  # Initial x position
        self.y = y  # Initial y position
        self.theta = theta  # Initial orientation (angle in radians)
        self.s = s  # Initial speed of the car

    def update_state(self, u_s, u_phi, dt):
        # Ensure u_s is within the bounds [-1, 1]
        u_s = max(min(u_s, 30), -30)
        
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

    def get_state(self):
        # Return the current state as (x, y, theta)
        return self.x, self.y, self.theta, self.s

    def accelerate_to_speed(self, target_speed, acceleration_rate, dt):
        """
        Accelerates the car to the target speed at a given rate.
        target_speed: Desired target speed (bounded by [-1, 1])
        acceleration_rate: Rate of speed increase per time step
        dt: Time step
        """
        # Ensure the target speed is within bounds [-1, 1]
        target_speed = max(min(target_speed, 30), -30)
        
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
def main():
    # Define car parameters
    L = 1.5  # Wheelbase of the car

    # Create a SimpleCarModel instance
    # car = SimpleCarModel(L)

    # Example action input (speed and steering angle)
    u_phi = 0.1 # Steering angle in radians (bounded between [-pi/2, pi/2])
    # dt = 0.1  # Time step of 100ms

    # Example of accelerating to a target speed
    target_speed = 24.8  # Target speed (bounded between [-1, 1])
    acceleration_rate = 15.5  # Speed increase per time step

    # for _ in range(20):  # Simulate for 20 steps
    #     car.accelerate_to_speed(target_speed, acceleration_rate, dt)
    #     car.update_state(car.s, u_phi, dt)
    #     state = car.get_state()
    #     print(f"Accelerating -> x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}, speed: {state[3]:.2f}")

    # # Example of decelerating to stop
    # deceleration_rate = 0.1  # Speed decrease per time step

    # for _ in range(10):  # Simulate for 10 steps
    #     car.decelerate_to_stop(deceleration_rate, dt)
    #     car.update_state(car.s, u_phi, dt)
    #     state = car.get_state()
    #     print(f"Decelerating -> x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}, speed: {state[3]:.2f}")


# import time
# def main():
    global x, P, actual_path, estimated_path
    # clock = pygame.time.Clock()
    running = True
    global dt
    tmin = time.perf_counter()
    t = tmin
    sim_duration = 10  # Simulate for 10 seconds
    mean_time = 0
    counter = 1
    while running and t-tmin < sim_duration:
        screen.fill((0, 0, 0))  # Clear screen with white background

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simulate the actual mouse position using the exponential * cosine movement
        z = simulated_mouse_position(t-tmin)#+np.random.randn()*0.001)
        dt = time.perf_counter() - t
        # car.accelerate_to_speed(target_speed, acceleration_rate, dt)
        # car.update_state(car.s, u_phi, dt)
        # state = car.get_state()
        # z = np.array([[state[0]],[state[1]]])
        print(z)
        # Kalman filter prediction and update
        x_pred, P_pred = kalman_predict(x, P)
        # draw_uncertainty_ellipse(screen, P_pred, x_pred)
        x, P = kalman_update(x_pred, P_pred, z)

        # Track the actual and estimated positions
        actual_path.append((int(z[0, 0]), int(z[1, 0])))
        # estimated_path.append((int(x[0, 0]), int(x[1, 0])))

        # mean_time = (mean_time * counter + dt) / (counter + 1)
        # counter +=1

        future_time = dt

        A_future = np.array([[1, 0, future_time, 0, 0, 0, 0],
                             [0, 1, 0, future_time, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, future_time, 0.5 * future_time**2],
                             [0, 0, 0, 0, 0, 1, future_time],
                             [0, 0, 0, 0, 0, 0, 1]])
        x_future = A_future @ x
        estimated_path.append(((x_future[0, 0]), (x_future[1, 0])))
        # Draw current 
        # pygame.draw.circle(screen, (0, 255, 0), (int(z[0, 0]), int(z[1, 0])), 1)  # Green for actual
        # pygame.draw.circle(screen, (255, 0, 0), (int(x[0, 0]), int(x[1, 0])), 5)  # Red for estimated
        # pygame.draw.circle(screen, (255,255,255),(int(x_future[0, 0]), int(x_future[1, 0])), 5)
        # pygame.draw.line(screen, (255, 255, 255), (int(z[0, 0]), int(z[1, 0]))/1550*800, (int(x[0, 0]), int(x[1, 0]))/1550*800, 5)
        # pygame.draw.line(screen, (255, 255, 255), (int(x[0, 0]), int(x[1, 0])), (int(x_future[0, 0]), int(x_future[1, 0])), 5)
         # Draw the uncertainty ellipse at the estimated position
        draw_paths(screen)

        # draw_uncertainty_ellipse(screen, P, x_future)

        pygame.display.flip()  # Update display
        t += dt
        # print(t)
        time.sleep(0.01)
        x = x_future
    # After the simulation, draw the paths
    draw_paths(screen)

    pygame.time.delay(5000)  # Keep display open for 5 seconds before quitting
    pygame.quit()
    dist = lambda x,y: np.sqrt((x[:,0] - y[:,0])**2 + (x[:,1] - y[:,1])**2)
    print(np.array(actual_path).shape)
    # plt.plot(np.array(actual_path)[5:,0],np.array(actual_path)[5:,1])
    # plt.plot(np.array(estimated_path)[5:,0],np.array(estimated_path)[5:,1])
    d = dist(np.array(actual_path),np.array(estimated_path))[5:]
    # d = (d - np.min(d)) / (np.max(d) - np.min(d)) * 300
    plt.plot(np.array(actual_path)[5:,0],d,2,c='red')
    plt.grid()
    # plt.xlim([400,800])
    plt.legend(['actual','estimated','error'])
    plt.show()
import matplotlib.pyplot as plt

# np.linalg.norm
def draw_paths(screen):
    """Draws the actual and estimated paths on the screen."""
    screen.fill((0, 0, 0))  # Clear the screen

    # Draw the actual path in green
    for i in range(1, len(actual_path)):
        pygame.draw.line(screen, (0, 255, 0), actual_path[i - 1], actual_path[i], 1)

    # Draw the estimated path in red
    for i in range(1, len(estimated_path)):
        pygame.draw.line(screen, (255, 0, 0), estimated_path[i - 1], estimated_path[i], 1)
    
    pygame.display.flip()

if __name__ == "__main__":
    main()
