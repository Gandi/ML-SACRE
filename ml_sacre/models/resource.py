class Resource:

    def __init__(self,
                 name,
                 unit,
                 requested_value,
                 coef_margin,
                 reward_reallocation,
                 reward_underallocation,
                 reward_requested_allocation,
                 reward_optimal_allocation):
        self.name = name
        self.unit = unit
        self.requested_value = requested_value

        self.coef_margin = coef_margin

        self.reward_reallocation = reward_reallocation
        self.reward_underallocation = reward_underallocation
        self.reward_requested_allocation = reward_requested_allocation
        self.reward_optimal_allocation = reward_optimal_allocation

        self.col_requested = f'{self.name} Requested ({self.unit})'
        self.col_used = f'{self.name} Used ({self.unit})'
        self.col_predicted = f'{self.name} Predicted ({self.unit})'
