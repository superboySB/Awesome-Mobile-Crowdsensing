import pygame
import imageio
import numpy as np

# 初始化 Pygame
pygame.display.init()

# 设置 Pygame 屏幕尺寸
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# 设定结束时间为10秒 (10000毫秒)
end_time = 10000

frames = []  # 保存所有帧的列表

running = True
start_time = pygame.time.get_ticks()

while running:
    elapsed_time = pygame.time.get_ticks() - start_time
    if elapsed_time > end_time:
        running = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 例如，绘制一个移动的矩形
    screen.fill((255, 255, 255))
    rect_x = (pygame.time.get_ticks() // 10) % width
    pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(rect_x, 150, 60, 60))
    pygame.display.flip()

    # 捕获 Pygame 屏幕的帧并添加到列表中
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frames.append(frame)

    clock.tick(20)

# 使用 imageio 将帧保存为 GIF
imageio.mimsave('output.gif', frames, duration=10)

pygame.quit()