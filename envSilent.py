import random

import gymnasium as gym
import math
import numpy as np
#from kivy.uix.widget import Widget

#from kivy.config import Config
#from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector

from PIL import Image as PILImage

class Car():

    def __init__(self):
        self.angle = 0.0
        self.rotation = 0.0

        #self.velocity_x = 0.0
        #self.velocity_y = 0.0
        self.velocity = Vector(0, 0)

        self.sensor1_x = 0.0
        self.sensor1_y = 0.0

        self.sensor2_x = 0.0
        self.sensor2_y = 0.0

        self.sensor3_x = 0.0
        self.sensor3_y = 0.0

        self.signal1 = 0.0
        self.signal2 = 0.0
        self.signal3 = 0.0

        self.center_x = 0.0
        self.center_y = 0.0


class Game():
    def __init__(self):
        self.car = Car()
        self.ball1 = Ball1()
        self.ball2 = Ball2()
        self.ball3 = Ball3()
        self.goalpost = GoalPost()


class Ball1():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


class Ball2():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


class Ball3():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

class GoalPost():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


class MapEnv(gym.Env):

    def __init__(self):

        self.canvas = Game()

        img = PILImage.open("./images/mask.png").convert('L')
        self.sand = np.asarray(img) / 255

        self.map_size = (self.sand.shape[0], self.sand.shape[1])

        self.window_size = (self.map_size[0] / 2, self.map_size[1] / 2)
        self.max_action = 5.0  # 5 degree turn left or right

        self.action_space = gym.spaces.Box(
            low=-1.0 * self.max_action,
            high=1.0 * self.max_action,
            dtype=np.float16
        )

        self.observations = ['car_signals', 'car_position', 'goal_position', 'orientation', 'distance', 'score']
        self.goal_index = 0
        # self.goal_positions = [(1400, 150), (2224, 1041), (118, 270)]

        self.goal_positions = [(1200, 1110), (1500, 1200), (1710, 1100), (1100, 1100), (800, 1000), (1400, 150),
                               (2224, 1041), (118, 270)]

        self.log = ''

        self.observation_space = self._get_observation_space()
        self.state = {}
        self.canvas.car.center_x, self.canvas.car.center_y = (1132, 1092)
        self.max_speed = 2.0
        self.follow_flag = False

    def _get_observation_space(self):
        low_signals = np.array([0.0, 0.0, 0.0])
        high_signals = np.array([1.0, 1.0, 1.0])
        low_orientation = self.max_action * np.array([-1.0, -1.0])
        high_orientation = self.max_action * np.array([1.0, 1.0])

        return gym.spaces.Box(
            low=np.concatenate([low_signals, low_orientation]),
            high=np.concatenate([high_signals, high_orientation]),
            shape=(5,)
        )

    def _get_observation(self):
        s1, s2, s3 = self.state.get('car_signals')
        return np.array([s1, s2, s3, self.state.get('orientation', 0), -1. * self.state.get('orientation', 0)])

    def get_signals(self):
        s1 = round(int(
            np.sum(
                self.sand[
                int(self.canvas.car.sensor1_x) - 10:int(self.canvas.car.sensor1_x) + 10,
                int(self.canvas.car.sensor1_y) - 10:int(self.canvas.car.sensor1_y) + 10
                ]
            )
        ) / 400., 3)

        s2 = round(int(
            np.sum(
                self.sand[
                int(self.canvas.car.sensor2_x) - 10:int(self.canvas.car.sensor2_x) + 10,
                int(self.canvas.car.sensor2_y) - 10:int(self.canvas.car.sensor2_y) + 10
                ]
            )
        ) / 400., 3)

        s3 = round(int(
            np.sum(
                self.sand[
                int(self.canvas.car.sensor3_x) - 10:int(self.canvas.car.sensor3_x) + 10,
                int(self.canvas.car.sensor3_y) - 10:int(self.canvas.car.sensor3_y) + 10
                ]
            )
        ) / 400., 3)

        return [s1, s2, s3]

    def get_goal_distance(self):
        goal = self.state.get('goal_position')
        distance = np.sqrt(
            (int(self.canvas.car.center_x) - goal[0]) ** 2 +
            (int(self.canvas.car.center_y) - goal[1]) ** 2)

        distance = round(distance, 2)
        return distance

    def get_goal_orientation(self):
        goal = self.state.get('goal_position')
        xx = goal[0] - int(self.canvas.car.center_x)
        yy = goal[1] - int(self.canvas.car.center_y)
        orientation = Vector(*self.canvas.car.velocity).angle((xx, yy)) / 180.0
        orientation = round(orientation, 3)

        return orientation

    def update_state(self):

        done = False
        distance = self.get_goal_distance()
        orientation = self.get_goal_orientation()

        self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3 = self.get_signals()
        # print('signals: ', self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3)

        longueur = self.map_size[0]
        largeur = self.map_size[1]

        if self.canvas.car.sensor1_x > longueur - 15 or self.canvas.car.sensor1_x < 15 or \
                self.canvas.car.sensor1_y > largeur - 15 or self.canvas.car.sensor1_y < 15:
            self.canvas.car.signal1 = 1.

        if self.canvas.car.sensor2_x > longueur - 15 or self.canvas.car.sensor2_x < 15 or \
                self.canvas.car.sensor2_y > largeur - 15 or self.canvas.car.sensor2_y < 15:
            self.canvas.car.signal2 = 1.

        if self.canvas.car.sensor3_x > longueur - 15 or self.canvas.car.sensor3_x < 15 or \
                self.canvas.car.sensor3_y > largeur - 15 or self.canvas.car.sensor3_y < 15:
            self.canvas.car.signal3 = 1.

        if self.sand[int(self.canvas.car.center_x), int(self.canvas.car.center_y)] > 0:
            # done = True
            reward = -1.0

        else:  # otherwise
            reward = 1.0

            if self.follow_flag:
                if distance < self.state.get('distance'):
                    reward = 0.6

        if distance < 25:
            self.goal_index = (self.goal_index + 1) % len(self.goal_positions)

            self.canvas.goalpost.x = self.goal_positions[self.goal_index][0]
            self.canvas.goalpost.y = self.goal_positions[self.goal_index][1]

            reward += 1.0
            # done = False

        if self.canvas.car.center_x < 30:
            self.canvas.car.center_x = 30
            reward = -10.0

        elif self.canvas.car.center_x > self.map_size[0] - 30:
            self.canvas.car.center_x = self.map_size[0] - 30
            reward = -10.0

        if self.canvas.car.center_y < 30:
            self.canvas.car.center_y = 30
            reward = -10.0

        elif self.canvas.car.center_y > self.map_size[1] - 30:
            self.canvas.car.center_y = self.map_size[1] - 30
            reward = -10.0

        reward = round(reward, 2)
        score = self.state['score'] + reward

        self.state.update(
            dict(
                car_signals=[self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3],
                car_position=(int(self.canvas.car.center_x), int(self.canvas.car.center_y)),
                goal_position=np.array(self.goal_positions[self.goal_index]),
                orientation=orientation,
                distance=distance,
                score=score
            )
        )

        print('Reward: ', reward)
        return self._get_observation(), reward, done, {}

    def move(self, rotation):

        if self.sand[int(self.canvas.car.center_x), int(self.canvas.car.center_y)] > 0:
            new_speed = 0.5
        else:
            new_speed = 2.  # self.max_speed*(1.0 - abs(rotation)/self.max_action)

        self.canvas.car.rotation = rotation

        self.canvas.car.angle = self.canvas.car.angle + self.canvas.car.rotation

        self.canvas.car.sensor1_x, self.canvas.car.sensor1_y  = Vector(30, 0).rotate(self.canvas.car.angle) + (self.canvas.car.center_x, self.canvas.car.center_y)
        self.canvas.car.sensor2_x, self.canvas.car.sensor2_y = Vector(30, 0).rotate((self.canvas.car.angle + 30) % 360) + (self.canvas.car.center_x, self.canvas.car.center_y)
        self.canvas.car.sensor3_x, self.canvas.car.sensor3_y = Vector(30, 0).rotate((self.canvas.car.angle - 30) % 360) + (self.canvas.car.center_x, self.canvas.car.center_y)

        self.canvas.ball1.center_x, self.canvas.ball1.center_y  = self.canvas.car.sensor1_x, self.canvas.car.sensor1_y
        self.canvas.ball2.center_x, self.canvas.ball2.center_y = self.canvas.car.sensor2_x, self.canvas.car.sensor2_y
        self.canvas.ball3.center_x, self.canvas.ball3.center_y = self.canvas.car.sensor3_x, self.canvas.car.sensor3_y

        self.canvas.car.center_x, self.canvas.car.center_y = Vector(new_speed, 0).rotate(self.canvas.car.angle) + (self.canvas.car.center_x, self.canvas.car.center_y)

        # return self.update_state()

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed', 0))
        print("---------RESET---------")

        car_idx = random.randint(0, len(self.goal_positions) - 1)
        self.canvas.car.center_x, self.canvas.car.center_y = self.goal_positions[car_idx]
        self.canvas.car.velocity = Vector(2, 0)

        self.goal_index = random.randint(0, len(self.goal_positions) - 1)

        if car_idx == self.goal_index:
            self.goal_index = (self.goal_index + 1) % len(self.goal_positions)

        self.canvas.goalpost.x, self.canvas.goalpost.y  = self.goal_positions[self.goal_index]

        self.state = dict(
            car_signals=self.get_signals(),
            car_position=(self.canvas.car.center_x, self.canvas.car.center_y),
            goal_position=np.array((self.canvas.goalpost.x, self.canvas.goalpost.y)),
            score=0.0
        )

        self.state.update(dict(
            orientation=self.get_goal_orientation(),
            distance=self.get_goal_distance()
        ))

        return self._get_observation()

    def step(self, action):

        self.move(action.item())
        obs, reward, done, _ = self.update_state()
        print(self.state)
        return obs, reward, done, _


'''
class MapEnv(gym.Env):

    def __init__(self):

        self.canvas = Game()

        img = PILImage.open("./images/mask.png").convert('L')
        self.sand = np.asarray(img) / 255

        self.map_size = (self.sand.shape[0], self.sand.shape[1])

        self.window_size = (self.map_size[0]/2, self.map_size[1]/2)
        self.max_action = 5.0  # 5 degree turn left or right

        self.action_space = gym.spaces.Box(
            low=-1.0*self.max_action,
            high=1.0*self.max_action,
            dtype=np.float16
        )

        self.observations = ['car_signals', 'car_position', 'goal_position', 'orientation', 'distance', 'score']
        self.goal_index = 0
        #self.goal_positions = [(1400, 150), (2224, 1041), (118, 270)]

        self.goal_positions = [(1200, 1110), (1500, 1200), (1710, 1100), (1100, 1100), (800, 1000)]

        self.log = ''

        self.observation_space = self._get_observation_space()
        self.state = {}
        self.canvas.car.x, self.canvas.car.y = (1132, 1092)

    def _get_observation_space(self):
        low_signals = np.array([0.0, 0.0, 0.0])
        high_signals = np.array([1.0, 1.0, 1.0])
        low_orientation = self.max_action * np.array([-1.0, -1.0])
        high_orientation = self.max_action * np.array([1.0, 1.0])

        return gym.spaces.Box(
            low=np.concatenate([low_signals, low_orientation]),
            high=np.concatenate([high_signals, high_orientation]),
            shape=(5, ),
            dtype=np.float16
        )


    def _get_observation(self):
        s1, s2, s3 = self.state.get('car_signals')
        return np.array([s1, s2, s3, self.state.get('orientation', 0), -1. * self.state.get('orientation', 0)])


    def get_signals(self):
        s1 = int(
            np.sum(
                self.sand[
                int(self.canvas.car.sensor1_x) - 10:int(self.canvas.car.sensor1_x) + 10,
                int(self.canvas.car.sensor1_y) - 10:int(self.canvas.car.sensor1_y) + 10
                ]
            )
        ) / 400.

        s2 = int(
            np.sum(
                self.sand[
                int(self.canvas.car.sensor2_x)-10:int(self.canvas.car.sensor2_x)+10,
                int(self.canvas.car.sensor2_y)-10:int(self.canvas.car.sensor2_y)+10
                ]
            )
        )/400.

        s3 = int(
            np.sum(
                self.sand[
                int(self.canvas.car.sensor3_x)-10:int(self.canvas.car.sensor3_x)+10,
                int(self.canvas.car.sensor3_y)-10:int(self.canvas.car.sensor3_y)+10
                ]
            )
        )/400.

        return [s1, s2, s3]


    def get_goal_distance(self):
        return np.sqrt(
            (self.canvas.car.x - self.goal_positions[self.goal_index][0]) ** 2 +
            (self.canvas.car.y - self.goal_positions[self.goal_index][1]) ** 2)


    def get_goal_orientation(self):
        xx = self.goal_positions[self.goal_index][0] - self.canvas.car.x
        yy = self.goal_positions[self.goal_index][1] - self.canvas.car.y
        orientation = Vector(*self.canvas.car.velocity).angle((xx, yy)) / 180.0

        return orientation


    def update_state(self):

        done = False
        distance = self.get_goal_distance()
        orientation = self.get_goal_orientation()

        self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3 = self.get_signals()
        print('signals: ', self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3)

        longueur = self.map_size[0]
        largeur = self.map_size[1]

        if self.canvas.car.sensor1_x > longueur - 10 or self.canvas.car.sensor1_x < 10 or \
                self.canvas.car.sensor1_y > largeur - 10 or self.canvas.car.sensor1_y < 10:
            self.canvas.car.signal1 = 10.

        if self.canvas.car.sensor2_x > longueur - 10 or self.canvas.car.sensor2_x < 10 or \
                self.canvas.car.sensor2_y > largeur - 10 or self.canvas.car.sensor2_y < 10:
            self.canvas.car.signal2 = 10.

        if self.canvas.car.sensor3_x > longueur - 10 or self.canvas.car.sensor3_x < 10 or \
                self.canvas.car.sensor3_y > largeur - 10 or self.canvas.car.sensor3_y < 10:
            self.canvas.car.signal3 = 10.


        if self.sand[int(self.canvas.car.x), int(self.canvas.car.y)] > 0:
            self.canvas.car.velocity = Vector(0.5, 0).rotate(self.canvas.car.angle)
            done = True
            reward = -1.0
            score = self.state['score'] + reward

            self.state.update(
                dict(
                    car_signals=[self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3],
                    orientation=orientation,
                    distance=distance,
                    score=score
                )
            )

            return self._get_observation(), reward, done, {}

        else:  # otherwise
            self.canvas.car.velocity = Vector(2, 0).rotate(self.canvas.car.angle)
            reward = -0.2

            if distance < self.state.get('distance'):
                reward = 0.1

            if distance < 25:
                self.goal_index = (self.goal_index + 1) % len(self.goal_positions)

                self.canvas.goalpost.x = self.goal_positions[self.goal_index][0]
                self.canvas.goalpost.y = self.goal_positions[self.goal_index][1]

                reward = 1.0
                done = False

            if self.canvas.car.x < 10:
                self.canvas.car.x = 10
                reward = -1.0

            elif self.canvas.car.x > self.map_size[0] - 10:
                self.canvas.car.x = self.map_size[0] - 10
                reward = -1.0

            if self.canvas.car.y < 10:
                self.canvas.car.y = 10
                reward = -1.0

            elif self.canvas.car.y > self.map_size[1] - 10:
                self.canvas.car.y = self.map_size[1] - 10
                reward = -1.0

            score = self.state['score'] + reward

            self.state.update(
                dict(
                    car_signals=[self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3],
                    car_position=(self.canvas.car.x, self.canvas.car.y),
                    goal_position=(self.goal_positions[self.goal_index]),
                    orientation=orientation,
                    distance=distance,
                    score=score
                )
            )

            return self._get_observation(), reward, done, {}


    def move(self, rotation):
        _vel = Vector(*self.canvas.car.velocity)

        self.canvas.car.x, self.canvas.car.y = _vel.x + self.canvas.car.x, _vel.y + self.canvas.car.y
        self.canvas.car.rotation = rotation

        self.canvas.car.angle = self.canvas.car.angle + self.canvas.car.rotation

        s1 = Vector(30, 0).rotate(self.canvas.car.angle)
        self.canvas.car.sensor1_x, self.canvas.car.sensor1_y = s1.x + self.canvas.car.x, s1.y + self.canvas.car.y

        s2 = Vector(30, 0).rotate((self.canvas.car.angle+30)%360)
        self.canvas.car.sensor2_x, self.canvas.car.sensor2_y = s2.x + self.canvas.car.x, s2.y + self.canvas.car.y

        s3 = Vector(30, 0).rotate((self.canvas.car.angle-30)%360)
        self.canvas.car.sensor3_x, self.canvas.car.sensor3_y = s3.x + self.canvas.car.x, s3.y + self.canvas.car.y

        self.canvas.ball1.x, self.canvas.ball1.y = self.canvas.car.sensor1_x, self.canvas.car.sensor1_y
        self.canvas.ball2.x, self.canvas.ball2.y = self.canvas.car.sensor2_x, self.canvas.car.sensor1_y
        self.canvas.ball3.x, self.canvas.ball3.y = self.canvas.car.sensor3_x, self.canvas.car.sensor1_y

        #return self.update_state()


    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed', 0))
        print("---------RESET---------")
        self.canvas.car.x, self.canvas.car.y  = (1132, 1092)
        self.canvas.car.velocity = Vector(2, 0)

        self.goal_index = 0

        self.canvas.goalpost.x, self.canvas.car.y = self.goal_positions[self.goal_index]

        self.state = dict(
            car_signals=self.get_signals(),
            car_position=(self.canvas.car.x, self.canvas.car.y),
            goal_position=(self.canvas.goalpost.x, self.canvas.goalpost.y),
            orientation=self.get_goal_orientation(),
            distance=self.get_goal_distance(),
            score=0.0
        )

        return self._get_observation()


    def step(self, action):
        print(1,
              self.state.get('goal_position')[0],
              self.state.get('goal_position')[1],
              self.state.get('distance'),
              int(self.canvas.car.x), int(self.canvas.car.y),
              self.sand[int(self.canvas.car.x), int(self.canvas.car.y)])

        self.move( action.item() )
        obs, reward, done, _ = self.update_state()
        return obs, reward, done, _
'''

