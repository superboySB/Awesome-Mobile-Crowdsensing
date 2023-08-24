import pygame
from pygame.locals import *
import sys
import numpy as np
import pickle
import scipy.spatial as spt
import paramiko
import os

WIDTH = 1400
HEIGHT = 650

FPS = 30

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
SCALE = 15

RED = (255, 0, 0, 120)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)

NUM_UAV = 3
UAV_COLOR = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255),
             (138, 43, 226), (255, 69, 0), (210, 180, 40)]


TASK_LIST = [[0, 0, 10, 10], [0, 10, 10, 20], [0, 20, 10, 30], [0, 30, 10, 40], [0, 40, 10, 50], [0, 50, 10, 60], [10, 0, 20, 10], [10, 10, 20, 20], [10, 20, 20, 30], [10, 30, 20, 40], [10, 40, 20, 50], [10, 50, 20, 60], [20, 0, 30, 10], [20, 10, 30, 20], [20, 20, 30, 30], [20, 30, 30, 40], [20, 40, 30, 50], [20, 50, 30, 60], [30, 0, 40, 10], [30, 10, 40, 20], [30, 20, 40, 30], [30, 30, 40, 40], [30, 40, 40, 50], [30, 50, 40, 60], [40, 0, 50, 10], [40, 10, 50, 20], [40, 20, 50, 30], [40, 30, 50, 40], [40, 40, 50, 50], [40, 50, 50, 60], [50, 0, 60, 10], [50, 10, 60, 20], [50, 20, 60, 30], [50, 30, 60, 40], [50, 40, 60, 50], [50, 50, 60, 60]]

TASK_ORDER = [2, 5, 0, 6]
ALLOCATED = False
RENDER_INCREASE = False
RENDER_AOI  = True
OBSTACLE= [[28.39,6.64,36.87,8.85],
                    [15.15,16.56,18.44,22.12],
                    [9.96,28.32,13.64,30.90],
                    [28.58,21.98,36.32,27.25],
                    [44.25,26.14,50.44,28.72],
                    [65.13,28.40,74.01,32.89],
                    [20.02,34.88,34.99,38.53]]
INDEX_CONVERT = []

UAV_TYPE =['carrier','uav']

class UAV(pygame.sprite.Sprite):
    # sprite for the Player
    def __init__(self, trajectory, screen, index=0):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.smoothscale(pygame.image.load("./img/drone.png"), [50, 50])
        self.rect = self.image.get_rect()
        self.screen = screen

        self.rect.center = (trajectory[0][0] * SCALE, trajectory[0][1] * SCALE)
        self.trajectory = trajectory
        self.step = 0
        self.font = pygame.font.Font(None, int(1.7 * SCALE))
        self.index = index

    def update(self):
        self.rect.x = self.trajectory[self.step][0] * SCALE
        self.rect.y = self.trajectory[self.step][1] * SCALE
        # print(self.rect)

        self.screen.blit(
            self.font.render('uav:{}'.format(self.index + 1), True, (0, 0, 0), (255, 255, 255)),
            (self.rect.x, self.rect.y))

        for j in range(self.step - 1):
            pygame.draw.line(self.screen, UAV_COLOR[self.index],
                             (self.trajectory[j][0] * SCALE, self.trajectory[j][1] * SCALE),
                             (self.trajectory[j + 1][0] * SCALE, self.trajectory[j + 1][1] * SCALE), 4)

        self.step += 1
        if self.step == len(self.trajectory):
            self.step = 0


class PoIs(pygame.sprite.Sprite):
    # sprite for the Player
    def __init__(self, info, screen,full_info):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.smoothscale( pygame.image.load("./poi.png"), [1, 1])
        self.rect = self.image.get_rect()
        self.screen = screen
        self.info = info
        self.step = 0
        self.radius = 0.125 * SCALE
        self.size = (0.5 * SCALE, 0.5 * SCALE)
        self.font = pygame.font.Font(None, 1 * SCALE)
        self.full_info = full_info
        if RENDER_INCREASE:
            self.poi_speed = full_info['poi_arrival']
        
        

    def update(self):

        for index in range(len(OBSTACLE)):
            x1,y1,x2,y2=OBSTACLE[index][0]*SCALE,OBSTACLE[index][1]*SCALE,OBSTACLE[index][2]*SCALE,OBSTACLE[index][3]*SCALE
            pygame.draw.polygon(self.screen, (255,0,0,255),
                                        [ [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
                                    )
        for index in range(len(self.info[self.step]['val'])):
            if self.info[self.step]['val'][index] > 0:
                color = (0, 0, 255, min(255, (self.info[self.step]['val'][index]) * 10.5))
                pygame.draw.circle(self.screen, color,
                                   (self.info[self.step]['pos'][index][0] * SCALE,
                                    self.info[self.step]['pos'][index][1] * SCALE),
                                   self.radius)
                
                if RENDER_INCREASE:
                    color = (255, 0,0, min(255, (self.poi_speed[index,self.step]) * 20.5))
                    pygame.draw.circle(self.screen, color,
                                   (self.info[self.step]['pos'][index][0] * SCALE,
                                    self.info[self.step]['pos'][index][1] * SCALE),
                                   self.radius+3)
                
                else:
                    if RENDER_AOI:
                         self.screen.blit(
                        self.font.render(str(round(self.info[self.step]['aoi'][index]/20, 1)), True, (0, 0, 0),
                                        (255, 255, 255)),
                        (self.info[self.step]['pos'][index][0] * SCALE, self.info[self.step]['pos'][index][1] * SCALE))
                    else:
                        self.screen.blit(self.font.render(str(round(self.info[self.step]['val'][index], 1)), True, (0, 0, 0),
                                        (255, 255, 255)),
                        (self.info[self.step]['pos'][index][0] * SCALE, self.info[self.step]['pos'][index][1] * SCALE))

            

        self.step += 1
        if self.step == len(self.info):
            self.step = 0

class Carrier(pygame.sprite.Sprite):
    # sprite for the Player
    def __init__(self, trajectory, screen, index=0):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.smoothscale(pygame.image.load("./img/carrier.png"), [60, 60])
        self.rect = self.image.get_rect()
        self.screen = screen

        self.rect.center = (trajectory[0][0] * SCALE, trajectory[0][1] * SCALE)
        self.trajectory = trajectory
        self.step = 0
        self.font = pygame.font.Font(None, int(1.7 * SCALE))
        self.index = index

    def update(self):
        self.rect.x = self.trajectory[self.step][0] * SCALE
        self.rect.y = self.trajectory[self.step][1] * SCALE
        # print(self.rect)

        self.screen.blit(
            self.font.render('uav:{}'.format(self.index + 1), True, (0, 0, 0), (255, 255, 255)),
            (self.rect.x, self.rect.y))

        for j in range(self.step - 1):
            pygame.draw.line(self.screen, UAV_COLOR[self.index],
                             (self.trajectory[j][0] * SCALE, self.trajectory[j][1] * SCALE),
                             (self.trajectory[j + 1][0] * SCALE, self.trajectory[j + 1][1] * SCALE), 4)

        self.step += 1
        if self.step == len(self.trajectory):
            self.step = 0


class RenderEnv(object):

    def __init__(self, info):
        pygame.init()
        pygame.display.set_caption('Virtual MCS')
        self.info = info
        self.clock = pygame.time.Clock()
        self.windows = pygame.display.set_mode((WIDTH, HEIGHT))
        self.screen = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.all = pygame.sprite.LayeredUpdates()
        self.font = pygame.font.Font(None, int(2 * SCALE))
        self.small_font = pygame.font.Font(None, int(1.5 * SCALE))
        self.step = 0
        self.total_reward = {key:0 for key in UAV_TYPE}
        self.render_allocated = ALLOCATED
        self.task_allocated = None
        # if self.render_allocated:
        # self.task_allocated = info['selected_area']

        self.NUM_STEP = len(info['poi_history'])
        self.NUM_UAV = {}
        for type in UAV_TYPE:
            self.NUM_UAV[type] = len(self.info[type]['uav_trace'])

        self.init_sprite()

    def init_sprite(self):
        self.PoIs = PoIs(self.info['poi_history'], self.screen,self.info)
        self.all.add(self.PoIs)
        for i in range(self.NUM_UAV['uav']):
            uav = UAV(self.info['uav']['uav_trace'][i], self.screen, i)
            self.all.add(uav)
        for i in range(self.NUM_UAV['carrier']):
            uav = Carrier(self.info['carrier']['uav_trace'][i], self.screen, i)
            self.all.add(uav)
        
    def start(self):
        Rendering = True
        paused = False

        while Rendering:
            self.windows.fill(WHITE)
            self.screen.fill(WHITE)
            self.clock.tick(FPS)
            # Process input (events)

            for event in pygame.event.get():
                # check for closing window
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        pygame.quit()
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if not paused:
                # Update
                self.all.update()

                # Draw / render
                self.all.draw(self.screen)
                # *after* drawing everything, flip the display
                self.windows.blit(self.screen, (10 * SCALE, 0))
                self.render_info()
                self.step += 1
                if self.step == self.NUM_STEP:
                    self.total_reward = {key:0 for key in UAV_TYPE}
                    self.step = 0
                pygame.display.update()

    def render_info(self):
        txt = []
        txt.append(self.font.render('step:' + str(self.step), True, (0, 0, 0), (255, 255, 255)))

        for type in UAV_TYPE:
            reward = self.info[type]['reward_history'][self.step]
            self.total_reward[type] += reward
            txt.append(
                self.font.render(type+'_uav_reward:{}'.format(round(reward, 2)), True, (0, 0, 0),
                                    (255, 255, 255)))
            txt.append(
                self.font.render(type+'_total:{}'.format(round(self.total_reward[type], 2)), True, (0, 0, 0),
                                    (255, 255, 255)))

        for index, t in enumerate(txt):
            self.windows.blit(t, (0, index * 2 * SCALE))


# def get_convex_hull():
#     config = Config()
#     poi_position = np.array(config("poi_position"))
#     task_position = np.array(config("task_position"))
#     allocated = []
#     for group_index in range(poi_position.shape[0]):
#         g = []
#         hull = spt.ConvexHull(points=poi_position[group_index, :, :])
#         for vertice in hull.vertices:
#             g.append(vertice)
#         allocated.append(poi_position[group_index, g, :] * SCALE * config("map_x"))
#     return allocated


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


if __name__ == '__main__':

    path = '/home/liuchi/fangchen/AirDropMCS/LaunchMCS/default_1.txt'
    # INDEX_CONVERT = get_convex_hull()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("10.1.114.77", 22, "liuchi", "LIUCHI-linc-2021!", timeout=5)
    sftp_client = client.open_sftp()
    remote_file = sftp_client.open(path, 'r')
    info = pickle.load(remote_file)
    remote_file.close()
    # path = '/Users/wanghao/wh/Code/Python/mcs_dynamically/manager_1653191380.txt'
    # info = load_variavle(path)
    # if 'manager' in path:
    #     ALLOCATED = True
    R = RenderEnv(info)
    R.start()
