import gymnasium as gym
import math, random
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

        self.goal_positions = [(1200, 1110), (1500, 1200), (1710, 1100), (1100, 1100), (800, 1000), (1400, 150), (2224, 1041), (118, 270)]

        self.log = ''

        self.observation_space = self._get_observation_space()
        self.state = {}
        self.canvas.car.center = (1132, 1092)
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
            shape=(5, )
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
                int(self.canvas.car.sensor2_x)-10:int(self.canvas.car.sensor2_x)+10,
                int(self.canvas.car.sensor2_y)-10:int(self.canvas.car.sensor2_y)+10
                ]
            )
        )/400., 3)

        s3 = round(int(
            np.sum(
                self.sand[
                int(self.canvas.car.sensor3_x)-10:int(self.canvas.car.sensor3_x)+10,
                int(self.canvas.car.sensor3_y)-10:int(self.canvas.car.sensor3_y)+10
                ]
            )
        )/400., 3)

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
        xx =  goal[0] - int(self.canvas.car.center_x)
        yy = goal[1] - int(self.canvas.car.center_y)
        orientation = Vector(*self.canvas.car.velocity).angle((xx, yy)) / 180.0
        orientation = round(orientation, 3)

        return orientation


    def update_state(self):

        done = False
        distance = self.get_goal_distance()
        orientation = self.get_goal_orientation()

        self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3 = self.get_signals()
        #print('signals: ', self.canvas.car.signal1, self.canvas.car.signal2, self.canvas.car.signal3)

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
            #done = True
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
            #done = False

        if self.canvas.car.x < 20:
            self.canvas.car.x = 20
            reward = -10.0

        elif self.canvas.car.x > self.map_size[0] - 20:
            self.canvas.car.x = self.map_size[0] - 20
            reward = -10.0

        if self.canvas.car.y < 20:
            self.canvas.car.y = 20
            reward = -10.0

        elif self.canvas.car.y > self.map_size[1] - 20:
            self.canvas.car.y = self.map_size[1] - 20
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
            new_speed = 2. #self.max_speed*(1.0 - abs(rotation)/self.max_action)

        self.canvas.car.rotation = rotation

        self.canvas.car.angle = self.canvas.car.angle + self.canvas.car.rotation

        self.canvas.car.sensor1 = Vector(30, 0).rotate(self.canvas.car.angle) + self.canvas.car.center
        self.canvas.car.sensor2 = Vector(30, 0).rotate((self.canvas.car.angle+30)%360) + self.canvas.car.center
        self.canvas.car.sensor3 = Vector(30, 0).rotate((self.canvas.car.angle-30)%360) + self.canvas.car.center

        self.canvas.ball1.center = self.canvas.car.sensor1
        self.canvas.ball2.center = self.canvas.car.sensor2
        self.canvas.ball3.center = self.canvas.car.sensor3

        self.canvas.car.center = Vector(new_speed, 0).rotate(self.canvas.car.angle) + self.canvas.car.center

        #return self.update_state()


    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed', 0))
        print("---------RESET---------")

        car_idx = random.randint(0, len(self.goal_positions)-1)
        self.canvas.car.center = self.goal_positions[car_idx]
        self.canvas.car.velocity = Vector(2, 0)

        self.goal_index = random.randint(0, len(self.goal_positions)-1)

        if car_idx == self.goal_index:
            self.goal_index = (self.goal_index+1)%len(self.goal_positions)

        self.canvas.goalpost.pos = self.goal_positions[self.goal_index]

        self.state = dict(
            car_signals=self.get_signals(),
            car_position=self.canvas.car.center,
            goal_position=np.array(self.canvas.goalpost.pos),
            score=0.0
        )
        
        self.state.update(dict(
            orientation=self.get_goal_orientation(),
            distance=self.get_goal_distance()
        ))

        return self._get_observation()


    def step(self, action):

        self.move( action.item() )
        obs, reward, done, _ = self.update_state()
        print(self.state)
        return obs, reward, done, _


