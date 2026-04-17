# 2x2 Panel Chart Summary

Selected layers:

- 0, 3, 17, 28

Selection rule:

- layer 0 is included explicitly
- one early representative layer with the highest average oracle-baseline logit-MSE delta
- one middle representative layer with the highest average oracle-baseline logit-MSE delta
- one late representative layer with the highest average oracle-baseline logit-MSE delta

Chosen example per layer:

- layer 0: task `hotpotqa`, example `139`, bits `4`
- layer 3: task `qasper`, example `44`, bits `2`
- layer 17: task `hotpotqa`, example `139`, bits `2`
- layer 28: task `qasper`, example `44`, bits `2`