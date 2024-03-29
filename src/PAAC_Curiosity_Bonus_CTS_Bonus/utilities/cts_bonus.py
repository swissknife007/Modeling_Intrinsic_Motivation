# credit: https://github.com/brendanator/atari-rl/blob/master/agents/exploration_bonus.py
import cv2
from skimage.transform import resize

import math
import numpy as np

from utilities.cts import CTS
# import util


class ExplorationBonus(object):
  def __init__(self, width=21, height=21, num_bins=4, beta=0.05):
    self.frame_shape = (width, height)
    self.beta = beta
    self.density_model = CTS(context_length=4, max_alphabet_size=num_bins)

  def bonus(self, observation):
    # Get 3-bit frame
    frame = cv2.resize(observation[-1], self.frame_shape) // 32
    # frame = resize(observation, (42, 42), mode='constant')

    # Calculate pseudo count
    prob = self.update_density_model(frame)
    recoding_prob = self.density_model_probability(frame)
    pseudo_count = prob * (1 - recoding_prob) / (recoding_prob - prob)
    if pseudo_count < 0:
      pseudo_count = 0  # Occasionally happens at start of training

    # Return exploration bonus
    exploration_bonus = self.beta / math.sqrt(pseudo_count + 0.01)
    return exploration_bonus

  def update_density_model(self, frame):
    return self.sum_pixel_probabilities(frame, self.density_model.update)

  def density_model_probability(self, frame):
    return self.sum_pixel_probabilities(frame, self.density_model.log_prob)

  def sum_pixel_probabilities(self, frame, log_prob_func):
    total_log_probability = 0.0

    for y in range(frame.shape[0]):
      for x in range(frame.shape[1]):
        context = self.context(frame, y, x)
        pixel = frame[y, x]
        total_log_probability += log_prob_func(context=context, symbol=pixel)

    return math.exp(total_log_probability)

  def context(self, frame, y, x):
    """This grabs the L-shaped context around a given pixel"""

    OUT_OF_BOUNDS = 7
    context = [OUT_OF_BOUNDS] * 4

    if x > 0:
      context[3] = frame[y][x - 1]

    if y > 0:
      context[2] = frame[y - 1][x]

      if x > 0:
        context[1] = frame[y - 1][x - 1]

      if x < frame.shape[1] - 1:
        context[0] = frame[y - 1][x + 1]

    # The most important context symbol, 'left', comes last.
    return context
