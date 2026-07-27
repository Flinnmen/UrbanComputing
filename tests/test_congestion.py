import unittest

import numpy as np

from skymonitor.congestion import get_congestion_level_change


class TestCongestionLevelChange(unittest.TestCase):
	def test_level_change_uses_current_to_next_step(self):
		levels = np.array([[[1], [4], [4]]], dtype=np.int8)

		changed, weights = get_congestion_level_change(levels, weights=(1.0, 2.0, 3.0))

		self.assertEqual(changed.tolist(), [[[True], [False], [False]]])
		self.assertEqual(weights.tolist(), [[[6.0], [0.0], [0.0]]])


if __name__ == '__main__':
	unittest.main()
