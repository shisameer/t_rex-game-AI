import pygame
import random
import dino_game  # your original game file
from dino_game import (
    Dinosaur, SmallCactus, LargeCactus, Bird, Cloud,
    BG, SCREEN, SCREEN_WIDTH, SCREEN_HEIGHT
)

class Game:
    """
    Wrapper for the T-Rex runner to support programmatic stepping (for NEAT training).
    """
    ACTION_JUMP = 0
    ACTION_DUCK = 1
    ACTION_NONE = 2

    def __init__(self, speed=20):
        pygame.init()
        self.display = SCREEN
        self.clock = pygame.time.Clock()
        self.base_speed = speed
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.reset()

    def reset(self):
        """Initialize or reset game state."""
        self.player = Dinosaur()
        self.cloud = Cloud()
        self.obstacles = []
        self.speed = self.base_speed
        self.bg_x = 0
        self.bg_y = 380
        self.points = 0
        self.game_over = False
        # sync global for background and obstacles
        dino_game.game_speed = self.speed

    def get_state(self):
        """Return normalized inputs for the neural network."""
        # Distance to next obstacle
        if self.obstacles:
            obs = self.obstacles[0]
            dist = (obs.rect.x - self.player.dino_rect.x) / SCREEN_WIDTH
            obs_h = (SCREEN_HEIGHT - obs.rect.y) / SCREEN_HEIGHT
        else:
            dist, obs_h = 1.0, 0.0
        vel = self.player.jump_vel / self.player.JUMP_VEL
        speed_norm = self.speed / 50.0
        return [dist, obs_h, vel, speed_norm]

    def step(self, action):
        """
        Advance one frame given an action:
          - action: 0=jump, 1=duck, 2=do nothing
        Returns:
          state: next state vector
          reward: points gained this frame
          done: whether the dino has crashed
        """
        # set global speed
        self.speed = self.speed
        dino_game.game_speed = self.speed

        # apply action
        if action == self.ACTION_JUMP and not self.player.dino_jump:
            self.player.dino_jump = True
            self.player.dino_run = False
            self.player.dino_duck = False
        elif action == self.ACTION_DUCK and not self.player.dino_jump:
            self.player.dino_duck = True
            self.player.dino_run = False
            self.player.dino_jump = False
        else:
            if not self.player.dino_jump:
                self.player.dino_duck = False
                self.player.dino_run = True

        # update player (no real keyboard input)
        fake_input = {pygame.K_UP: False, pygame.K_DOWN: False}
        self.player.update(fake_input)

        # spawn obstacles if none
        if not self.obstacles:
            choice = random.randint(0, 2)
            if choice == 0:
                self.obstacles.append(SmallCactus(dino_game.SMALL_CACTUS))
            elif choice == 1:
                self.obstacles.append(LargeCactus(dino_game.LARGE_CACTUS))
            else:
                self.obstacles.append(Bird(dino_game.BIRD))

        # update obstacles
        for obs in list(self.obstacles):
            obs.update()
            if obs.rect.x < -obs.rect.width:
                self.obstacles.remove(obs)

        # update background
        width = BG.get_width()
        self.bg_x -= self.speed
        if self.bg_x <= -width:
            self.bg_x = 0

        # update cloud
        self.cloud.update()

        # update score and speed
        self.points += 1
        reward = 1
        if self.points % 100 == 0:
            self.speed += 1
            dino_game.game_speed = self.speed

        # check for collision
        for obs in self.obstacles:
            if self.player.dino_rect.colliderect(obs.rect):
                self.game_over = True

        # return next state, reward, done flag
        return self.get_state(), reward, self.game_over

    def render(self):
        """Draw current state to the pygame display (if you need visualization)."""
        # background
        SCREEN.blit(BG, (self.bg_x, self.bg_y))
        SCREEN.blit(BG, (self.bg_x + BG.get_width(), self.bg_y))
        # cloud, obstacles, player
        self.cloud.draw(self.display)
        for obs in self.obstacles:
            obs.draw(self.display)
        self.player.draw(self.display)
        # score display
        text = self.font.render(f"Points: {self.points}", True, (0,0,0))
        self.display.blit(text, (1000, 40))
        pygame.display.update()
        self.clock.tick(30)
