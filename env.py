import gymnasium as gym
import math
import numpy as np
from kivy.uix.widget import Widget

from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector

from PIL import Image as PILImage

class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)


class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    goalpost = ObjectProperty(None)


class Ball1(Widget):
    pass

class Ball2(Widget):
    pass

class Ball3(Widget):
    pass

class GoalPost(Widget):
    pass


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
        self.canvas.car.pos = (1132, 1092)

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
                    car_position=self.canvas.car.pos,
                    goal_position=np.array(self.goal_positions[self.goal_index]),
                    orientation=orientation,
                    distance=distance,
                    score=score
                )
            )

            return self._get_observation(), reward, done, {}


    def move(self, rotation):
        self.canvas.car.pos = Vector(*self.canvas.car.velocity) + self.canvas.car.pos
        self.canvas.car.rotation = rotation

        self.canvas.car.angle = self.canvas.car.angle + self.canvas.car.rotation

        self.canvas.car.sensor1 = Vector(30, 0).rotate(self.canvas.car.angle) + self.canvas.car.pos
        self.canvas.car.sensor2 = Vector(30, 0).rotate((self.canvas.car.angle+30)%360) + self.canvas.car.pos
        self.canvas.car.sensor3 = Vector(30, 0).rotate((self.canvas.car.angle-30)%360) + self.canvas.car.pos

        self.canvas.ball1.pos = self.canvas.car.sensor1
        self.canvas.ball2.pos = self.canvas.car.sensor2
        self.canvas.ball3.pos = self.canvas.car.sensor3

        #return self.update_state()


    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed', 0))
        print("---------RESET---------")
        self.canvas.car.center = (1132, 1092)
        self.canvas.car.velocity = Vector(2, 0)

        self.goal_index = 0

        self.canvas.goalpost.pos = self.goal_positions[self.goal_index]

        self.state = dict(
            car_signals=self.get_signals(),
            car_position=self.canvas.car.center,
            goal_position=np.array(self.canvas.goalpost.pos),
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


