import math

class SimpleCarModel:
    def __init__(self, L, min_phi=-math.pi/2, max_phi=math.pi/2, x=0.0, y=0.0, theta=0.0, s=0.0):
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
        u_s = max(min(u_s, 1), -1)
        
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
        target_speed = max(min(target_speed, 1), -1)
        
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

# Example of usage:
if __name__ == "__main__":
    # Define car parameters
    L = 0.5  # Wheelbase of the car

    # Create a SimpleCarModel instance
    car = SimpleCarModel(L)

    # Example action input (speed and steering angle)
    u_phi = 0.5  # Steering angle in radians (bounded between [-pi/2, pi/2])
    dt = 0.1  # Time step of 100ms

    # Example of accelerating to a target speed
    target_speed = 0.8  # Target speed (bounded between [-1, 1])
    acceleration_rate = 0.05  # Speed increase per time step

    for _ in range(20):  # Simulate for 20 steps
        car.accelerate_to_speed(target_speed, acceleration_rate, dt)
        car.update_state(car.s, u_phi, dt)
        state = car.get_state()
        print(f"Accelerating -> x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}, speed: {state[3]:.2f}")

    # Example of decelerating to stop
    deceleration_rate = 0.1  # Speed decrease per time step

    for _ in range(10):  # Simulate for 10 steps
        car.decelerate_to_stop(deceleration_rate, dt)
        car.update_state(car.s, u_phi, dt)
        state = car.get_state()
        print(f"Decelerating -> x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}, speed: {state[3]:.2f}")
