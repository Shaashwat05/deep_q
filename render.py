from tensorflow.python.keras.models import load_model
import gym
import numpy as np

env = gym.make('CartPole-v1')


model = load_model("cp_model.h5")

for i in range(20):
    state = env.reset()
    done  = False
    while not done:
        env.render()
        action = np.argmax(model.predict(np.array([state])))
        new_state, reward, done, observation = env.step(action)
        state = new_state