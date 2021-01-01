import pygame
import sys
import random
from collections import deque
import neat

class FlappyBirdEnv:

    def __init__(self, size=(400, 600), gravity=0.25, frame_rate=60, population=10):

        self.SIZE = size
        self.GRAVITY = gravity 
        self.FRAME_RATE = frame_rate
        self.observation_shape = 2
        self.action_shape = 2
        self.bird_velocities = [0 for _ in range(population)]
        self.jump_velocity = -4
        self.TB_PIPE_GAP = int(0.25 * self.SIZE[1])
        self.SS_PIPE_GAP = self.SIZE[0] // 2
        self.GROUNDY = int(0.85 * self.SIZE[1])

        self.pipe_moving_freq = 3
        self.pipe_list = deque(maxlen=4)

        self.image_counter = 0
        self.population = population

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

        # Birds surface
        
        self.bird_surface = pygame.image.load('assets/bluebird-midflap.png').convert_alpha()
        self.bird_rects = []
        self.isAlive = []
        self.start_bird_pos = [int(0.2 * self.SIZE[0]), int(0.425 * self.SIZE[1])]

        for _ in range(self.population):
            bird_rect = self.bird_surface.get_rect(center=(self.start_bird_pos[0], random.randint(int(self.SIZE[1] * 0.1), int(0.8 * self.SIZE[1]))))
            self.bird_rects.append(bird_rect) 
            self.isAlive.append(True)       

        # Pipes
        self.pipe_surface = pygame.image.load('assets/pipe-green.png').convert()
        self.bottom_pipe_list = deque(maxlen=4)
        self.top_pipe_list = deque(maxlen=4)
        
        # Font
        self.font = pygame.font.Font('freesansbold.ttf', 16)

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

    
    def check_collision(self, bird_index):

        for pipe in self.bottom_pipe_list:
            if self.bird_rects[bird_index].colliderect(pipe):
                return True, 'PIPE'
        
        for pipe in self.top_pipe_list:
            if self.bird_rects[bird_index].colliderect(pipe):
                return True, 'PIPE'

        if self.bird_rects[bird_index].top <= 0:
            return True, 'TOP'
        
        if self.bird_rects[bird_index].bottom >=  int(0.85 * self.SIZE[1]):
            return True, 'BOTTOM'
        
        return False, None


    def draw_floor(self):

        self.screen.blit(self.floor_surface, (self.floor_x_pos, self.GROUNDY))
        self.screen.blit(self.floor_surface, (self.floor_x_pos + self.SIZE[0], int(0.85 * self.SIZE[1])))

        self.floor_x_pos -= 1
        if self.floor_x_pos <= -self.SIZE[0]:
            self.floor_x_pos = 0


    def draw_birds(self):

        for bird_index in range(self.population):
            if self.isAlive[bird_index]:
                rotated_bird = pygame.transform.rotozoom(self.bird_surface, - self.bird_velocities[bird_index] * 5, 1)
                self.screen.blit(rotated_bird, self.bird_rects[bird_index])


    def draw_pipes(self):

        for pipe in self.bottom_pipe_list:

            self.screen.blit(self.pipe_surface, pipe)
        
        for pipe in self.top_pipe_list:

            flip_pipe = pygame.transform.flip(self.pipe_surface, False, True)
            self.screen.blit(flip_pipe, pipe)
    

    def get_observation(self, bird_index):

        # Observation state will be [height from ground, bird_velocity
        # height at which next gap is there, horizontal distance between next pipe and bird]

        observation = [
            self.bird_rects[bird_index].centery - self.pipe_list[0][1] + random.random(),
            self.pipe_list[0][0] - self.bird_rects[bird_index].centerx + random.random()]
        
        return observation

    
    def move_bird(self, bird_index):
        self.bird_rects[bird_index].centery += self.bird_velocities[bird_index]


    def move_pipes(self):

        for i in range(len(self.bottom_pipe_list)):

            self.bottom_pipe_list[i].centerx -= self.pipe_moving_freq
            self.top_pipe_list[i].centerx -= self.pipe_moving_freq
        
        for i in range(len(self.pipe_list)):

            self.pipe_list[i][0] -= self.pipe_moving_freq

        if self.pipe_list[0][0] <= self.bird_rects[0].centerx:
            self.pipe_list.popleft()
            return True
        
        return False
    
    def reset(self):

        self.bird_velocity = 0.
        self.bottom_pipe_list.clear()
        self.top_pipe_list.clear()
        self.pipe_list.clear()
        for bird_index in range(self.population):
            self.bird_rects[bird_index].centery = self.start_bird_pos[1]
        self.add_pipe()

        return self.get_observation()


    def sample_action(self):

        return random.choice([0, 1])
    
    def draw(self, generation, image_counter):

        ret = self.move_pipes()
        self.add_pipe()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.bird_velocity = self.jump_velocity
            
        self.screen.blit(self.bg_surface, (0, 0))
        self.draw_birds()
        self.draw_pipes()
        self.draw_floor()

        text = self.font.render(f"Generation : {generation}", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.SIZE[0] * 0.5, self.SIZE[1] * 0.3))
        self.screen.blit(text, text_rect)

        pygame.display.update()
        self.clock.tick(self.FRAME_RATE)

        pygame.image.save(self.screen, f"images/{image_counter}.png")
    
        return ret, any(self.isAlive)


    def step(self, action, bird_index):


        if action == 1:
            self.bird_velocities[bird_index] = self.jump_velocity
        elif action == 0:
            pass
        else:
            raise ValueError(f"Got unexpected action - {action}")
        
        self.bird_velocities[bird_index] += self.GRAVITY
        self.move_bird(bird_index)
            

        is_collided, place_of_collision = self.check_collision(bird_index)
        if is_collided:
            self.isAlive[bird_index] = False

        return is_collided


def main(genomes, config):

    global generation, env, image_counter
    generation += 1
    env = FlappyBirdEnv(population=10)
    
    score = 0
    start_time = pygame.time.get_ticks()

    models_list = []
    genomes_list = []

    for genome_id, genome in genomes:

        genome.fitness = 0
        genomes_list.append(genome)
        model = neat.nn.FeedForwardNetwork.create(genome, config)
        models_list.append(model)
    
    run = True

    while run:

        game_time = round((pygame.time.get_ticks() - start_time)/1000, 2)
        ret, run = env.draw(generation, image_counter)
        image_counter += 1
        for bird_index in range(env.population):
            if env.isAlive[bird_index]:
                network_input = env.get_observation(bird_index)
                output = models_list[bird_index].activate(network_input)

                action = 0 if output[0] > 0.5 else 1
                is_collided = env.step(action, bird_index) 

                genomes_list[bird_index].fitness = game_time + score - is_collided * 10


def run_NEAT(config_filename):

    config = neat.config.Config(neat.DefaultGenome, 
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, 
                                neat.DefaultStagnation,
                                config_filename)
    
    neat_pop = neat.population.Population(config)
    
    neat_pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    neat_pop.add_reporter(stats)
    
    neat_pop.run(main, 15)
    
global generation
global image_counter
generation = 0
image_counter = 0

run_NEAT('config_file.txt')