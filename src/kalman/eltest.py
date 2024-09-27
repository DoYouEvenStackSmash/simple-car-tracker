import numpy as np
import pygame

def draw_uncertainty_ellipse(screen, P, x_est, screen_size):
    """Draws the uncertainty ellipse and its axes based on the covariance matrix P and the estimated state x."""
    # Extract the covariance submatrix for position (2x2 block)
    cov = P[:2, :2]

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
    scale_factor = 0.2 * max_screen_dim / (2 * np.sqrt(eigenvalues[0]))  # Scales according to largest eigenvalue

    # Scale the axes for visualization
    width = 2 * a * scale_factor  # Major axis length
    height = 2 * b * scale_factor  # Minor axis length

    # Ellipse angle from the eigenvector direction
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi

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

def random_covariance_matrix():
    """Generates a random symmetric positive-definite 2x2 covariance matrix."""
    A = np.random.randn(2, 2)
    return np.dot(A, A.T)  # P = A * A^T ensures positive-definiteness

def test_random_ellipses(screen, screen_size, num_ellipses=5):
    """Test the drawing of multiple random uncertainty ellipses with generated covariance matrices."""
    for _ in range(num_ellipses):
        # Generate a random covariance matrix
        P = random_covariance_matrix()

        # Random position for the ellipse
        x_est = np.array([[np.random.randint(100, screen_size[0] - 100)],
                          [np.random.randint(100, screen_size[1] - 100)]])

        # Draw the random ellipse
        draw_uncertainty_ellipse(screen, P, x_est, screen_size)

# Example usage in a Pygame loop
def main():
    pygame.init()
    screen_size = (800, 600)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Random Uncertainty Ellipses")
    clock = pygame.time.Clock()

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Clear screen with white background

        # Draw random ellipses each frame
        test_random_ellipses(screen, screen_size, num_ellipses=3)

        pygame.display.flip()  # Update the display
        clock.tick(1)  # Limit to 1 FPS for better visualization

    pygame.quit()

if __name__ == "__main__":
    main()
