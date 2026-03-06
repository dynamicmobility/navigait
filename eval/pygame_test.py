import pygame
import sys

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller found!")
    sys.exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

WIDTH, HEIGHT = 400, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PS4 Controller Visualizer")
font = pygame.font.SysFont(None, 30)

running = True
while running:
    screen.fill((30, 30, 30))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw joystick axes
    lx = joystick.get_axis(0)  # Left stick X
    ly = joystick.get_axis(1)  # Left stick Y
    rx = joystick.get_axis(2)  # Right stick X
    ry = joystick.get_axis(3)  # Right stick Y
    print(lx)

    # Left stick visualization
    pygame.draw.circle(screen, (200, 200, 200), (100, 150), 40, 2)
    pygame.draw.circle(screen, (0, 255, 0), (100 + int(lx * 40), 150 + int(ly * 40)), 10)

    # Right stick visualization
    pygame.draw.circle(screen, (200, 200, 200), (300, 150), 40, 2)
    pygame.draw.circle(screen, (0, 0, 255), (300 + int(rx * 40), 150 + int(ry * 40)), 10)

    # Show pressed buttons
    pressed_buttons = [i for i in range(joystick.get_numbuttons()) if joystick.get_button(i)]
    text = font.render(f"Buttons pressed: {pressed_buttons}", True, (255, 255, 255))
    screen.blit(text, (20, 20))

    pygame.display.flip()
    pygame.time.wait(20)

pygame.quit()