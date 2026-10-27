"""Both methods observe Adaptive exactly once after each fresh first update."""

from gstar_opd_formal.controllers.adaptive import AdaptiveHorizonPolicy


class FreshPolicy:
    def __init__(self, settings: object) -> None:
        self.adaptive = AdaptiveHorizonPolicy(
            settings.protocol.horizons.choices, settings.adaptive
        )
        self.fresh_observations = 0

    @property
    def horizon(self) -> int:
        return self.adaptive.current_horizon

    def observe(self, observation: object) -> None:
        self.adaptive.observe(observation)
        self.fresh_observations += 1

    def state_dict(self) -> dict:
        return {
            "fresh_observations": self.fresh_observations,
            "adaptive": self.adaptive.state_dict(),
        }

    def load_state_dict(self, value: dict) -> None:
        count = value.get("fresh_observations")
        if type(count) is not int or count < 0:
            raise ValueError("Invalid fresh observation count")
        self.adaptive.load_state_dict(value["adaptive"])
        self.fresh_observations = count
