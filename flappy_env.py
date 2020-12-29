import pygame
import sys
import random
from collections import deque

class FlappyBirdEnv:

    def __init__(self, size=(400, 600), gravity=0.25, frame_rate=60, render=False):

        self.SIZE = size
        self.GRAVITY = gravity 
        self.FRAME_RATE = frame_rate
        self.observation_shape = 2
        self.action_shape = 2
        self.bird_velocity = 0.
        self.jump_velocity = -3
        self.TB_PIPE_GAP = int(0.25 * self.SIZE[1])
        self.SS_PIPE_GAP = self.SIZE[0] // 2
        self.GROUNDY = int(0.85 * self.SIZE[1])

        self.pipe_moving_freq = 3
        self.pipe_list = deque(maxlen=4)

        self.step_size = 4
        self.image_counter = 0

        self.render = render
        if self.render:

            pygame.init()
            self.screen = pygame.display.set_mode(self.SIZE)
            self.clock = pygame.time.Clock()
            
            # Background Surface
            self.bg_surface = pygame.image.load('assets/background-day.png').convert()
            self.bg_surface = pygame.transform.scale(self.bg_surface, self.SIZE)

            # Floor surface
            self.floor_surface = pygame.image.load('assets/base.png').convert()
            self.floor_surface = pygame.transform.scale(self.floor_surface, (self.SIZE[0], self.floor_surface.get_height()))
            self.floor_x_pos = 0

            # Bird surface
            self.bird_surface = pygame.image.load('assets/bluebird-midflap.png').convert_alpha()
            self.start_bird_pos = [int(0.2 * self.SIZE[0]), int(0.425 * self.SIZE[1])]
            self.bird_rect = self.bird_surface.get_rect(center=self.start_bird_pos)

            # Pipes
            self.pipe_surface = pygame.image.load('assets/pipe-green.png').convert()
            self.bottom_pipe_list = deque(maxlen=4)
            self.top_pipe_list = deque(maxlen=4)

        else:        
            # Bird surface
            self.bird_surface = pygame.image.load('assets/bluebird-midflap.png')
            self.start_bird_pos = [int(0.2 * self.SIZE[0]), int(0.425 * self.SIZE[1])]
            self.bird_rect = self.bird_surface.get_rect(center=self.start_bird_pos)

            # Pipes
            self.pipe_surface = pygame.image.load('assets/pipe-green.png')
            self.bottom_pipe_list = deque(maxlen=4)
            self.top_pipe_list = deque(maxlen=4)
        
        self.add_pipe()


    def add_pipe(self):
    
        if (not self.bottom_pipe_list) or (int(self.SIZE[0] + self.pipe_surface.get_width() // 2) - self.bottom_pipe_list[-1].centerx >= self.SS_PIPE_GAP):

            random_pipe_pos = random.randint(int(0.4 * self.SIZE[1]), int(0.75 * self.SIZE[1]))

            bottom_pipepos = [int(self.SIZE[0] + self.pipe_surface.get_width() // 2), random_pipe_pos]
            top_pipepos = [int(self.SIZE[0] + self.pipe_surface.get_width() // 2), random_pipe_pos - self.TB_PIPE_GAP]

            bottom_pipe = self.pipe_surface.get_rect(midtop=bottom_pipepos)
            top_pipe = self.pipe_surface.get_rect(midbottom=top_pipepos)
        
            self.bottom_pipe_list.append(bottom_pipe)
            self.top_pipe_list.append(top_pipe)
            self.pipe_list.append([int(self.SIZE[0] + self.pipe_surface.get_width() // 2), random_pipe_pos - (self.TB_PIPE_GAP // 2)])

    
    def check_collision(self):

        for pipe in self.bottom_pipe_list:
            if self.bird_rect.colliderect(pipe):
                return True, 'PIPE'
        
        for pipe in self.top_pipe_list:
            if self.bird_rect.colliderect(pipe):
                return True, 'PIPE'

        if self.bird_rect.top <= 0:
            return True, 'TOP'
        
        if self.bird_rect.bottom >=  int(0.85 * self.SIZE[1]):
            return True, 'BOTTOM'
        
        return False, None


    def draw_floor(self):

        self.screen.blit(self.floor_surface, (self.floor_x_pos, self.GROUNDY))
        self.screen.blit(self.floor_surface, (self.floor_x_pos + self.SIZE[0], int(0.85 * self.SIZE[1])))

        self.floor_x_pos -= 1
        if self.floor_x_pos <= -self.SIZE[0]:
            self.floor_x_pos = 0


    def draw_bird(self):

        rotated_bird = pygame.transform.rotozoom(self.bird_surface, - self.bird_velocity * 5, 1)
        self.screen.blit(rotated_bird, self.bird_rect)


    def draw_pipes(self):

        for pipe in self.bottom_pipe_list:

            self.screen.blit(self.pipe_surface, pipe)
        
        for pipe in self.top_pipe_list:

            flip_pipe = pygame.transform.flip(self.pipe_surface, False, True)
            self.screen.blit(flip_pipe, pipe)
    

    def get_observation(self):

        # Observation state will be [height from ground, bird_velocity
        # height at which next gap is there, horizontal distance between next pipe and bird]

        observation = [
            self.bird_rect.centery - self.pipe_list[0][1] + random.random(),
            self.pipe_list[0][0] - self.bird_rect.centerx + random.random()]
        
        return observation

    
    def move_bird(self):
        self.bird_rect.centery += self.bird_velocity


    def move_pipes(self):

        for i in range(len(self.bottom_pipe_list)):

            self.bottom_pipe_list[i].centerx -= self.pipe_moving_freq
            self.top_pipe_list[i].centerx -= self.pipe_moving_freq
        
        for i in range(len(self.pipe_list)):

            self.pipe_list[i][0] -= self.pipe_moving_freq

        if self.pipe_list[0][0] <= self.bird_rect.centerx:
            self.pipe_list.popleft()
            return True
        
        return False
    
    def reset(self):

        self.bird_velocity = 0.
        self.bottom_pipe_list.clear()
        self.top_pipe_list.clear()
        self.pipe_list.clear()
        self.bird_rect.centery = self.start_bird_pos[1]
        self.add_pipe()

        return self.get_observation()


    def sample_action(self):

        return random.choice([0, 1])

    def step(self, action):

        actions = [action] + [0] * self.step_size
        reward = -0.02

        for action in actions:
            if action == 1:
                self.bird_velocity = self.jump_velocity
            elif action == 0:
                pass
            else:
                raise ValueError(f"Got unexpected action - {action}")
            
            self.bird_velocity += self.GRAVITY
            self.move_bird()
            ret = self.move_pipes()
            self.add_pipe()

            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            self.bird_velocity = self.jump_velocity
            
                self.screen.blit(self.bg_surface, (0, 0))
                self.draw_bird()
                self.draw_pipes()
                self.draw_floor()

                pygame.display.update()
                self.clock.tick(self.FRAME_RATE)

                pygame.image.save(self.screen, f"images/{self.image_counter}.png")
                self.image_counter += 1

            is_collided, place_of_collision = self.check_collision()

            if is_collided:
                if place_of_collision == 'TOP':
                    reward = -5
                elif place_of_collision == 'BOTTOM':
                    reward = -5
                elif place_of_collision == 'PIPE':
                    reward = -2
            elif ret:
                reward = 10


        return self.get_observation(), reward, is_collided

