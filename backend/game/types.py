from __future__ import annotations

from enum import IntEnum


class PlayerType(IntEnum):
	HUMAN = 0
	RANDOM = 1
	BASELINE = 2
	ISMCTS = 3
	DMC = 4

	@property
	def name(self):
		return {
			PlayerType.HUMAN: "Human",
			PlayerType.RANDOM: "Random Bot",
			PlayerType.BASELINE: "Baseline Bot",
			PlayerType.ISMCTS: "ISMCTS Bot",
			PlayerType.DMC: "DMC Bot",
		}[self]
